# app.py
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import fitz
import logging
from pathlib import Path
from datetime import datetime
import uuid
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import re
from typing import List
from flask_cors import CORS

# ---------- App setup ----------
app = Flask(__name__)
CORS(app)

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

load_dotenv()

# In-memory sessions (non-prod)
sessions = {}

# ---------- API Key manager ----------
class APIKeyManager:
    REQUIRED_KEYS = ['PINECONE_API_KEY', 'GROQ_API_KEY']
    
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
            # Log but do not crash (non-prod)
            logger.warning(f"Missing required API keys: {', '.join(missing_keys)}")
        return keys

# Validate API keys on startup (non-fatal)
APIKeyManager.load_and_validate()

# ---------- Text processing ----------
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
        return text.encode("ascii", "ignore").decode().strip()
    
    @staticmethod
    def create_chunks(text: str) -> List[str]:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=100, 
            length_function=len, 
            separators=["\n\n", "\n", " ", ""]
        )
        return text_splitter.split_text(text)

# ---------- PDF processing ----------
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
            doc.close()
            return " ".join(text_parts) if text_parts else ""
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}", exc_info=True)
            return ""

# ---------- Vector store helpers ----------
class VectorStore:
    @staticmethod
    def initialize():
        """
        Initialize HuggingFace embeddings + Pinecone index and wrap with LangChain PineconeVectorStore.
        This mirrors your original implementation but wrapped in a function for reuse.
        """
        try:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/bert-base-nli-mean-tokens",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True},
                cache_folder="./models"
            )
            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            index = pc.Index("ragshi")
            return PineconeVectorStore(index=index, embedding=embeddings, text_key="text")
        except Exception as e:
            logger.error(f"Vector store init failed: {e}", exc_info=True)
            raise

    @staticmethod
    def add_texts(vector_store, texts, session_id):
        metadatas = [{"session_id": session_id} for _ in texts]
        ids = [f"{session_id}_{i}" for i in range(len(texts))]
        vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    
    @staticmethod
    def cleanup_session(session_id):
        if not session_id:
            return
        try:
            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            index = pc.Index("ragshi")
            # Try to query by filter; if pinecone requires a vector, we wrap this in try/except.
            try:
                query_response = index.query(
                    vector=[0.0] * 768,
                    filter={"session_id": session_id},
                    top_k=10000
                )
                if getattr(query_response, "matches", None):
                    vector_ids = [match.id for match in query_response.matches]
                    index.delete(ids=vector_ids)
            except Exception as inner_e:
                logger.warning(f"Could not query index for session cleanup (dimension or api mismatch): {inner_e}", exc_info=True)
        except Exception as e:
            logger.warning(f"Pinecone cleanup issue: {e}", exc_info=True)

# ---------- QA chain ----------
class QAChain:
    def __init__(self):
        # Keep model selection consistent with your code
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"), 
            model_name="llama-3.3-70b-versatile"
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions based on the provided context.
            If the answer cannot be found in the context, say "I couldn't find any relevant information in the documents."
            Context: {context}"""),
            ("human", "{question}")
        ])
    
    def get_response(self, question: str, session_id: str, vector_store) -> str:
        retriever = vector_store.as_retriever(
            search_kwargs={"filter": {"session_id": session_id}, "k": 4}
        )
        docs = retriever.invoke(question)
        if not docs:
            return "I couldn't find any relevant information in the documents."
        
        qa_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return qa_chain.invoke(question)

# ---------- Minimal non-prod session creation (drop-in fix) ----------
def get_or_create_session(session_id=None):
    """
    Minimal, non-prod version:
    - If client provided a session_id and it exists -> return it.
    - If client provided a session_id and it doesn't exist -> create session using that exact id.
    - If no session_id provided -> create a new random id.
    Keeps sessions in-memory (no persistence) — this is intentionally minimal.
    """
    # If existing -> return
    if session_id and session_id in sessions:
        return sessions[session_id], session_id

    # Use provided id if present, else create a new uuid
    chosen_id = session_id if session_id else str(uuid.uuid4())

    sessions[chosen_id] = {
        'chat_history': [],
        'processed_files': set(),
        'vector_store': None,
        'session_id': chosen_id
    }

    logger.info(f"Created new session: {chosen_id} (client_provided_id={bool(session_id)})")
    return sessions[chosen_id], chosen_id

# ---------- API endpoints ----------
@app.route('/api/upload', methods=['POST'])
def upload_files():
    try:
        session_id = request.form.get('session_id', '')
        session_data, session_id = get_or_create_session(session_id)
        
        files = request.files.getlist('files')
        if not files:
            return jsonify({'error': 'No files uploaded'}), 400
        
        results = []
        pdf_processor = PDFProcessor()
        text_processor = TextProcessor()
        
        for file in files:
            try:
                if file.filename not in session_data['processed_files']:
                    logger.info(f"Processing file: {file.filename}")
                    text = pdf_processor.extract_text(file) if file.filename.lower().endswith('.pdf') else ""
                    
                    if text:
                        logger.info(f"Extracted {len(text)} characters from {file.filename}")
                        chunks = text_processor.create_chunks(text)
                        logger.info(f"Created {len(chunks)} chunks from {file.filename}")
                        
                        if not session_data['vector_store']:
                            try:
                                logger.info("Initializing vector store...")
                                session_data['vector_store'] = VectorStore.initialize()
                                logger.info("Vector store initialized")
                            except Exception as e:
                                logger.warning(f"Vector store initialization failed during upload: {e}", exc_info=True)
                                # leave vector_store as None - behavior remains non-prod
        
                        if session_data['vector_store']:
                            VectorStore.add_texts(session_data['vector_store'], chunks, session_id)
                        else:
                            # If vector store not available, we still mark processed to avoid reprocessing repeatedly
                            logger.warning("Vector store unavailable; chunks won't be added to vector DB this run.")
                        
                        session_data['processed_files'].add(file.filename)
                        results.append({'filename': file.filename, 'status': 'processed', 'chunks': len(chunks)})
                        logger.info(f"Successfully processed {file.filename}")
                    else:
                        logger.warning(f"No text extracted from {file.filename}")
                        results.append({'filename': file.filename, 'status': 'no text extracted'})
                else:
                    results.append({'filename': file.filename, 'status': 'already processed'})
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {str(e)}", exc_info=True)
                results.append({'filename': file.filename, 'status': 'error', 'error': str(e)})
        
        return jsonify({
            'results': results, 
            'session_id': session_id,
            'processed_files': list(session_data['processed_files']),
            'vector_store_initialized': session_data['vector_store'] is not None
        }), 200
    except Exception as e:
        logger.error(f"Upload endpoint error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/query', methods=['POST'])
def query_documents():
    try:
        data = request.get_json()
        question = data.get('question')
        session_id = data.get('session_id', '')
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        logger.info(f"Query received - Session ID: {session_id}, Question: {question}")
        session_data, session_id = get_or_create_session(session_id)
        
        logger.info(f"Vector store exists: {session_data['vector_store'] is not None}")
        logger.info(f"Processed files: {session_data['processed_files']}")
        
        if not session_data['vector_store']:
            return jsonify({
                'answer': "Please upload documents first before asking questions.",
                'chat_history': session_data['chat_history'],
                'session_id': session_id
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
    except Exception as e:
        logger.error(f"Query endpoint error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear', methods=['POST'])
def clear_session():
    data = request.get_json() or {}
    session_id = data.get('session_id', '')
    
    if session_id and session_id in sessions:
        VectorStore.cleanup_session(session_id)
        del sessions[session_id]
    
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
