from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import fitz
import logging
from pathlib import Path
from datetime import datetime
import uuid
import re
from typing import List
from flask_cors import CORS
from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# API Key Manager
class APIKeyManager:
    REQUIRED_KEYS = ['PINECONE_API_KEY', 'PINECONE_ENVIRONMENT', 'GROQ_API_KEY', 'HUGGINGFACE_API_KEY']
    
    @staticmethod
    def load_and_validate():
        keys = {}
        missing_keys = []
        for key_name in APIKeyManager.REQUIRED_KEYS:
            key_value = os.getenv(key_name)
            if not key_value:
                missing_keys.append(key_name)
            keys[key_name] = key_value
        if missing_keys:
            raise Exception(f"Missing required API keys: {', '.join(missing_keys)}")
        return keys

# Validate API keys on startup
try:
    APIKeyManager.load_and_validate()
except Exception as e:
    logger.error(f"API Key validation failed: {str(e)}")
    # Continue running to allow debugging, but log the error

# Session state dictionary to track sessions by ID
sessions = {}

class TextProcessor:
    @staticmethod
    def clean_text(text: str) -> str:
        cleaning_steps = [
            (r"[\[\{\<].*?[\]\}\>]", ""),
            (r"\s+", " "),
            ("\t", " "),
            (u'\xa0', ' '),
            (u'\u200b', '')
        ]
        for pattern, replacement in cleaning_steps:
            text = re.sub(pattern, replacement, text)
        cleaned = text.encode("ascii", "ignore").decode().strip()
        logger.info(f"Cleaned text length: {len(cleaned)}")
        return cleaned
    
    @staticmethod
    def create_chunks(text: str) -> List[str]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Increased chunk size
            chunk_overlap=50,  # Reduced overlap
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        logger.info(f"Created {len(chunks)} chunks")
        return chunks

class PDFProcessor:
    @staticmethod
    def extract_text(pdf_file) -> str:
        try:
            pdf_bytes = pdf_file.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text_parts = []
            for page_num in range(len(doc)):
                text = doc[page_num].get_text()
                if text.strip():
                    text_parts.append(f"Page {page_num + 1}:\n{text}")
                    logger.info(f"Extracted text from page {page_num + 1}, length: {len(text)}")
            final_text = " ".join(text_parts) if text_parts else ""
            logger.info(f"Total extracted text length: {len(final_text)}")
            return final_text
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            return ""

class VectorStore:
    @staticmethod
    def initialize():
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True},
            cache_folder="./models"
        )
        
        # Create a new index with the correct dimensions
        index_name = "rag-index"
        if index_name not in pc.list_indexes().names():
            logger.info("Creating new Pinecone index")
            pc.create_index(
                name=index_name,
                dimension=384,  # dimension for all-MiniLM-L6-v2
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        index = pc.Index(index_name)
        vector_store = PineconeVectorStore(index=index, embedding=embeddings, text_key="text")
        logger.info("Vector store initialized successfully")
        return vector_store
    
    @staticmethod
    def add_texts(vector_store, texts, session_id):
        logger.info(f"Adding {len(texts)} texts to vector store for session {session_id}")
        metadatas = [{"session_id": session_id} for _ in texts]
        ids = [f"{session_id}_{i}" for i in range(len(texts))]
        try:
            vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
            logger.info("Texts added successfully to vector store")
        except Exception as e:
            logger.error(f"Error adding texts to vector store: {str(e)}")
            raise
    
    @staticmethod
    def cleanup_session(session_id):
        if session_id:
            logger.info(f"Cleaning up session {session_id}")
            index = pc.Index("rag-index")
            try:
                # Delete all vectors with the session_id metadata
                delete_response = index.delete(
                    filter={"session_id": session_id}
                )
                logger.info(f"Deleted vectors for session {session_id}")
            except Exception as e:
                logger.error(f"Error cleaning up session {session_id}: {str(e)}")

class QAChain:
    def __init__(self):
        self.llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile")
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions based on the provided context.
            If the answer cannot be found in the context, say "I couldn't find any relevant information in the documents."
            Context: {context}"""),
            ("human", "{question}")
        ])
    
    def get_response(self, question: str, session_id: str, vector_store) -> str:
        logger.info(f"Querying documents for question: {question}")
        try:
            # First, let's verify the index exists and has data
            index = pc.Index("rag-index")
            stats = index.describe_index_stats()
            logger.info(f"Index stats: {stats}")
            
            # Always use session filter
            retriever = vector_store.as_retriever(
                search_kwargs={
                    "filter": {"session_id": session_id},
                    "k": 8
                }
            )
            docs = retriever.get_relevant_documents(question)
            logger.info(f"Retrieved {len(docs)} relevant documents for session {session_id}")
            
            if not docs:
                logger.warning(f"No relevant documents found for session {session_id}")
                return "I couldn't find any relevant information in the documents."
            
            # Log the content of retrieved documents
            for i, doc in enumerate(docs):
                logger.info(f"Document {i+1} content: {doc.page_content[:200]}...")
            
            qa_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
            response = qa_chain.invoke(question)
            logger.info(f"Generated response: {response[:200]}...")
            return response
        except Exception as e:
            logger.error(f"Error in get_response: {str(e)}")
            return f"An error occurred while processing your question: {str(e)}"

# Helper function to get or create session
def get_or_create_session(session_id=None):
    if not session_id or session_id not in sessions:
        new_id = str(uuid.uuid4())
        sessions[new_id] = {
            'chat_history': [],
            'processed_files': set(),
            'vector_store': None,
            'session_id': new_id
        }
        return sessions[new_id], new_id
    return sessions[session_id], session_id

@app.route('/api/upload', methods=['POST'])
def upload_files():
    # Get session ID from request or create a new one
    session_id = request.form.get('session_id', '')
    session_data, session_id = get_or_create_session(session_id)
    
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No files uploaded'}), 400
    
    # Clean up old session data before processing new files
    if session_data['processed_files']:
        logger.info(f"Cleaning up old session data for session {session_id}")
        VectorStore.cleanup_session(session_id)
        session_data['processed_files'] = set()
        session_data['vector_store'] = None
        session_data['chat_history'] = []  # Also clear chat history
    
    results = []
    pdf_processor = PDFProcessor()
    text_processor = TextProcessor()
    
    for file in files:
        if file.filename not in session_data['processed_files']:
            logger.info(f"Processing file: {file.filename}")
            text = pdf_processor.extract_text(file) if file.filename.lower().endswith('.pdf') else ""
            
            # Check if any meaningful text was extracted
            cleaned_text = text_processor.clean_text(text) if text else ""
            if not cleaned_text or len(cleaned_text.strip()) < 50:  # Minimum meaningful text length
                logger.warning(f"No meaningful text extracted from {file.filename}")
                results.append({
                    'filename': file.filename,
                    'status': 'no text extracted',
                    'message': 'No readable text found. Please ensure this is a text-based PDF file.'
                })
                continue
                
            chunks = text_processor.create_chunks(cleaned_text)
            if chunks:
                if not session_data['vector_store']:
                    logger.info("Initializing vector store")
                    session_data['vector_store'] = VectorStore.initialize()
                logger.info(f"Adding {len(chunks)} chunks to vector store")
                VectorStore.add_texts(session_data['vector_store'], chunks, session_id)
                session_data['processed_files'].add(file.filename)
                results.append({
                    'filename': file.filename,
                    'status': 'processed',
                    'chunks': len(chunks)
                })
            else:
                logger.warning(f"No chunks created from {file.filename}")
                results.append({
                    'filename': file.filename,
                    'status': 'no text extracted',
                    'message': 'Could not process the text content.'
                })
    
    return jsonify({
        'results': results, 
        'session_id': session_id,
        'processed_files': list(session_data['processed_files'])
    }), 200

@app.route('/api/query', methods=['POST'])
def query_documents():
    data = request.get_json()
    question = data.get('question')
    session_id = data.get('session_id', '')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    logger.info(f"Received query request for session {session_id}")
    session_data, session_id = get_or_create_session(session_id)
    
    if not session_data['vector_store']:
        logger.warning("No vector store initialized for this session")
        return jsonify({
            'answer': "Please upload documents first before asking questions.",
            'chat_history': session_data['chat_history']
        }), 400
    
    qa_chain = QAChain()
    response = qa_chain.get_response(question, session_id, session_data['vector_store'])
    session_data['chat_history'].append({'role': 'user', 'content': question})
    session_data['chat_history'].append({'role': 'assistant', 'content': response})
    
    return jsonify({
        'answer': response, 
        'chat_history': session_data['chat_history'],
        'session_id': session_id
    }), 200

@app.route('/api/clear', methods=['POST'])
def clear_session():
    data = request.get_json() or {}
    session_id = data.get('session_id', '')
    
    if session_id and session_id in sessions:
        VectorStore.cleanup_session(session_id)
        # Remove this session
        del sessions[session_id]
    
    # Create a new session
    new_session, new_id = get_or_create_session()
    
    return jsonify({
        'message': 'Session cleared', 
        'session_id': new_id,
        'processed_files': []
    }), 200

@app.route('/api/session', methods=['GET'])
def get_session():
    session_id = request.args.get('session_id', '')
    session_data, session_id = get_or_create_session(session_id)
    
    return jsonify({
        'session_id': session_id,
        'processed_files': list(session_data['processed_files']),
        'chat_history': session_data['chat_history']
    }), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
