import streamlit as st
# Must be the first Streamlit command
st.set_page_config(page_title="Question.io", page_icon=":shark:", layout="wide")

from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_pinecone import Pinecone
from pinecone import Pinecone as PineconeClient
import fitz
import logging
import traceback
import re
from pathlib import Path
import time
from datetime import datetime

# Configure logging with both file and console handlers
def setup_logging() -> None:
    """Configure logging to both file and console with proper formatting"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

logger = logging.getLogger(__name__)

class APIKeyManager:
    """Manages API keys and their validation"""
    
    REQUIRED_KEYS = [
        'PINECONE_API_KEY',
        'PINECONE_ENVIRONMENT',
        'GROQ_API_KEY',
        'HUGGINGFACE_API_KEY'
    ]
    
    @staticmethod
    def load_and_validate() -> Dict[str, str]:
        """Load and validate all required API keys"""
        load_dotenv()
        
        keys = {}
        missing_keys = []
        
        for key_name in APIKeyManager.REQUIRED_KEYS:
            key_value = os.getenv(key_name)
            if not key_value:
                missing_keys.append(key_name)
            keys[key_name] = key_value
            
        if missing_keys:
            error_msg = f"Missing required API keys: {', '.join(missing_keys)}"
            logger.error(error_msg)
            st.error(error_msg)
            st.stop()
            
        return keys

class TextProcessor:
    """Handles text cleaning and processing operations"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text content"""
        cleaning_steps = [
            (r"[\[\{\<].*?[\]\}\>]", ""),  # Remove brackets and their contents
            (r"\s+", " "),                  # Normalize whitespace
            ("\t", " "),                    # Replace tabs
            (u'\xa0', ' '),                 # Replace non-breaking spaces
            (u'\u200b', '')                 # Remove zero-width spaces
        ]
        
        for pattern, replacement in cleaning_steps:
            text = re.sub(pattern, replacement, text)
            
        return text.encode("ascii", "ignore").decode().strip()
    
    @staticmethod
    def create_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        splitter = CharacterTextSplitter(
            separator=" ",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        return splitter.split_text(text)

class PDFProcessor:
    """Handles PDF document processing"""
    
    @staticmethod
    def extract_text(pdf_docs: List[Any]) -> str:
        """Extract text from multiple PDF documents"""
        combined_text = []
        
        for pdf in pdf_docs:
            try:
                # Try PyPDF2 first
                text = PDFProcessor._extract_with_pypdf2(pdf)
                if not text.strip():
                    # If PyPDF2 fails to extract meaningful text, try PyMuPDF
                    text = PDFProcessor._extract_with_pymupdf(pdf)
                combined_text.append(text)
            except Exception as e:
                logger.error(f"Failed to process PDF {pdf.name}: {str(e)}")
                st.error(f"Error processing {pdf.name}. Please check if the file is corrupted.")
                continue
                
        return " ".join(combined_text)
    
    @staticmethod
    def _extract_with_pypdf2(pdf: Any) -> str:
        """Extract text using PyPDF2"""
        pdf_reader = PdfReader(pdf)
        return " ".join(page.extract_text() for page in pdf_reader.pages)
    
    @staticmethod
    def _extract_with_pymupdf(pdf: Any) -> str:
        """Extract text using PyMuPDF"""
        pdf.seek(0)
        doc = fitz.open(stream=pdf.read(), filetype="pdf")
        text = " ".join(page.get_text() for page in doc)
        doc.close()
        return text

class VectorStoreManager:
    """Manages vector store operations"""
    
    def __init__(self, api_key: str, environment: str):
        self.api_key = api_key
        self.environment = environment
        self.client = PineconeClient(api_key=api_key)
    
    def initialize_store(self, text_chunks: List[str], index_name: str = "ragshi") -> Pinecone:
        """Initialize and populate vector store"""
        if not text_chunks:
            raise ValueError("No text chunks provided for vector store initialization")
            
        embeddings = HuggingFaceEmbeddings(
            model_name="all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        if index_name not in self.client.list_indexes().names():
            raise ValueError(f"Index '{index_name}' does not exist in Pinecone")
            
        vector_store = Pinecone.from_existing_index(
            index_name=index_name,
            embedding=embeddings,
            namespace="example-namespace"
        )
        
        # Add texts with retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                vector_store.add_texts(text_chunks)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"Retry {attempt + 1}/{max_retries} failed: {str(e)}")
                time.sleep(2 ** attempt)  # Exponential backoff
                
        return vector_store

class QAChain:
    """Manages the question-answering chain"""
    
    def __init__(self, vector_store: Pinecone, api_key: str):
        self.vector_store = vector_store
        self.api_key = api_key
        
    def create_chain(self) -> RetrievalQA:
        """Create and configure the QA chain"""
        llm = ChatGroq(
            model="mixtral-8x7b-32768",
            temperature=0.5,
            max_tokens=500,
            groq_api_key=self.api_key
        )
        
        retriever = self.vector_store.as_retriever(
            search_kwargs={
                "k": 2,
                
            }
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a knowledgeable assistant providing detailed answers based on the given documents.
            Always provide comprehensive explanations and cite specific parts of the documents when possible.
            If you're unsure about something or if the information isn't in the documents, be honest about it."""),
            ("human", "Context:\n{context}\n\nQuestion: {question}")
        ])
        
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={
                "prompt": prompt,
                "verbose": True
            }
        )

class StreamlitUI:
    """Manages the Streamlit user interface"""
    
    def __init__(self):
        self.initialize_session_state()
        
    @staticmethod
    def initialize_session_state() -> None:
        """Initialize session state variables"""
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "processed_files" not in st.session_state:
            st.session_state.processed_files = set()
    
    def render_header(self):
       
        st.title("Question.io")
            
    def render_sidebar(self) -> Optional[List[Any]]:  # Restored render_sidebar
        """Render the sidebar with file upload functionality."""
        with st.sidebar:
            st.title("Document Upload")
            pdf_docs = st.file_uploader(
                "Upload your PDFs",
                type=["pdf"],
                accept_multiple_files=True,
                help="Upload one or more PDF documents to analyze"
            )

            if pdf_docs:
                new_files = [
                    doc for doc in pdf_docs if doc.name not in st.session_state.processed_files
                ]

                if new_files:
                    if st.button("Process New Documents"):
                        return new_files

                elif pdf_docs and not new_files and st.session_state.processed_files:
                    st.write("All uploaded documents have been processed. Please upload new files.")

        return None
    
    def display_chat_history(self) -> None:
        """Display chat history with styled bubbles and scrolling."""
        if st.session_state.chat_history:
            for role, message in st.session_state.chat_history:
                with st.container():
                    message_class = "user-message" if role == "human" else "ai-message"
                    st.markdown(
                        f"""
                        <div class="{message_class}">
                            <div class="message-content">
                                {message}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.markdown(
                """
                <style>
                .user-message, .ai-message {
                    background-color: #36454F; /* Light gray for both */
                    padding: 10px;
                    margin-bottom: 5px;
                    border-radius: 10px; /* Rounded corners */
                    box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2); /* Subtle shadow */
                    max-width: 80%; /* Limit message width */
                    float: right; /* User messages to the right */
                    clear: both; /* Clear floats */
                }

                .ai-message {
                    float: left; /* AI messages to the left */
                }

                .message-content {
                    word-wrap: break-word;
                }

                .chat-container {
                    overflow-y: auto;
                    max-height: 500px;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            # Optional: Chat Container with Scrolling (as before)
            st.markdown("<div class='chat-container'>", unsafe_allow_html=True)  # Open container
            for role, message in st.session_state.chat_history: # Put the messages in the container
                # ... (message display code from above)
                pass # Replace with message display code
            st.markdown("</div>", unsafe_allow_html=True)  # Close container

            # Optional: Auto-scroll (as before)
            js = f"""
                <script>
                    const chatContainer = document.querySelector('.chat-container');
                    if (chatContainer) {{
                        chatContainer.scrollTop = chatContainer.scrollHeight;
                    }}
                </script>
            """
            st.components.v1.html(js, height=0)

def main():
    try:
        # Setup logging
        setup_logging()
        logger.info("Starting application")
        
        # Initialize UI
        ui = StreamlitUI()
        ui.render_header()
        
        # Load API keys
        keys = APIKeyManager.load_and_validate()
        
        # Initialize vector store manager
        vector_store_manager = VectorStoreManager(
            keys['PINECONE_API_KEY'],
            keys['PINECONE_ENVIRONMENT']
        )
        
        # Handle document processing
        ui = StreamlitUI()  # Create the StreamlitUI instance
       
    
        new_docs = ui.render_sidebar()  # Get the returned value (new files or None)
        if new_docs:  # Check if new_docs is NOT None (meaning there are new files)
         with st.spinner("Processing new documents..."):
            
                try:
                    # Process PDFs
                    raw_text = PDFProcessor.extract_text(new_docs)
                    cleaned_text = TextProcessor.clean_text(raw_text)
                    text_chunks = TextProcessor.create_chunks(cleaned_text)
                    
                    # Initialize or update vector store
                    vector_store = vector_store_manager.initialize_store(text_chunks)
                    st.session_state.vector_store = vector_store
                    
                    # Update processed files
                    st.session_state.processed_files.update(doc.name for doc in new_docs)
                    st.success("Documents processed successfully!")
                    
                except Exception as e:
                    logger.error(f"Document processing failed: {str(e)}")
                    logger.error(traceback.format_exc())
                    st.error("Failed to process documents. Please check the logs for details.")
                    return
        
        # Chat interface
        query = st.text_input(
            "Ask a question about your documents:",
            help="Enter your question here and click 'Send' to get an answer"
        )
        
        if query and st.button("Send"):
            if "vector_store" not in st.session_state:
                st.warning("Please upload and process documents first.")
                return
                
            with st.spinner("Searching for answer..."):
                try:
                    # Create QA chain
                    qa_chain = QAChain(
                        st.session_state.vector_store,
                        keys['GROQ_API_KEY']
                    ).create_chain()
                    
                    # Get response
                    response = qa_chain({"query": query})
                    
                    # Display response
                    st.markdown("### Response:")
                    st.write(response['result'])
                    
                    # Update chat history
                    st.session_state.chat_history.append(("human", query))
                    st.session_state.chat_history.append(("ai", response['result']))
                    
                    # Display updated chat history
                    ui.display_chat_history()
                    
                except Exception as e:
                    logger.error(f"Question processing failed: {str(e)}")
                    logger.error(traceback.format_exc())
                    st.error("Failed to process your question. Please try again.")
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        logger.error(traceback.format_exc())
        st.error("An unexpected error occurred. Please check the logs for details.")

if __name__ == "__main__":
    main()