from flask import Flask, request, jsonify
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import os
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
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

app = Flask(__name__)

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

APIKeyManager.load_and_validate()

# Session state
session_state = {
    'chat_history': [],
    'processed_files': set(),
    'vector_store': None,
    'session_id': str(uuid.uuid4())
}

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
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=100, length_function=len, separators=["\n\n", "\n", " ", ""]
        )
        return text_splitter.split_text(text)

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
            return " ".join(text_parts) if text_parts else ""
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            return ""

class VectorStore:
    @staticmethod
    def initialize():
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
    
    @staticmethod
    def add_texts(vector_store, texts, session_id):
        metadatas = [{"session_id": session_id} for _ in texts]
        ids = [f"{session_id}_{i}" for i in range(len(texts))]
        vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    
    @staticmethod
    def cleanup_session():
        if session_state['session_id']:
            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            index = pc.Index("ragshi")
            query_response = index.query(
                vector=[0.0] * 768, filter={"session_id": session_state['session_id']}, top_k=10000
            )
            if query_response.matches:
                vector_ids = [match.id for match in query_response.matches]
                index.delete(ids=vector_ids)
            session_state['session_id'] = str(uuid.uuid4())

class QAChain:
    def __init__(self):
        self.llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name="mixtral-8x7b-32768")
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions based on the provided context.
            If the answer cannot be found in the context, say "I couldn't find any relevant information in the documents."
            Context: {context}"""),
            ("human", "{question}")
        ])
    
    def get_response(self, question: str, session_id: str, vector_store) -> str:
        retriever = vector_store.as_retriever(search_kwargs={"filter": {"session_id": session_id}, "k": 4})
        docs = retriever.get_relevant_documents(question)
        if not docs:
            return "I couldn't find any relevant information in the documents."
        qa_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return qa_chain.invoke(question)

@app.route('/api/upload', methods=['POST'])
def upload_files():
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No files uploaded'}), 400
    
    results = []
    pdf_processor = PDFProcessor()
    text_processor = TextProcessor()
    
    for file in files:
        if file.filename not in session_state['processed_files']:
            text = pdf_processor.extract_text(file) if file.mimetype == 'application/pdf' else ""
            if text:
                chunks = text_processor.create_chunks(text)
                if not session_state['vector_store']:
                    session_state['vector_store'] = VectorStore.initialize()
                VectorStore.add_texts(session_state['vector_store'], chunks, session_state['session_id'])
                session_state['processed_files'].add(file.filename)
                results.append({'filename': file.filename, 'status': 'processed'})
            else:
                results.append({'filename': file.filename, 'status': 'no text extracted'})
    
    return jsonify({'results': results, 'session_id': session_state['session_id']}), 200

@app.route('/api/query', methods=['POST'])
def query_documents():
    data = request.get_json()
    question = data.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    qa_chain = QAChain()
    response = qa_chain.get_response(question, session_state['session_id'], session_state['vector_store'])
    session_state['chat_history'].append(('human', question))
    session_state['chat_history'].append(('assistant', response))
    
    return jsonify({'answer': response, 'chat_history': session_state['chat_history']}), 200

@app.route('/api/clear', methods=['POST'])
def clear_session():
    VectorStore.cleanup_session()
    session_state['processed_files'] = set()
    session_state['chat_history'] = []
    return jsonify({'message': 'Session cleared', 'session_id': session_state['session_id']}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)