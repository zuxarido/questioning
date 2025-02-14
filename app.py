import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import os
from nomic import embed
from langchain.embeddings.base import Embeddings
from typing import List
import fitz  # PyMuPDF
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
import requests

# Load environment variables
load_dotenv()

# Nomic API Key
NOMIC_API_KEY = os.getenv('NOMIC_API_KEY')
if not NOMIC_API_KEY:
    st.error("NOMIC_API_KEY not found in environment variables. Please set it in your .env file.")
    st.stop()

# Custom NomicEmbeddings class
class NomicEmbeddings(Embeddings):
    def __init__(self, model_name: str = "nomic-embed-text-v1.5", task_type: str = "search_document"):
        self.model_name = model_name
        self.task_type = task_type
        self.api_key = os.getenv('NOMIC_API_KEY')
        if not self.api_key:
            st.error("NOMIC_API_KEY not found in environment variables. Please set it in your .env file.")
            st.stop()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            response = embed.text(
                texts=texts,
                model=self.model_name,
                task_type=self.task_type
            )
            return response
        except Exception as e:
            st.error(f"Error embedding documents: {e}")
            return []

    def embed_query(self, text: str) -> List[float]:
        try:
            response = embed.text(
                texts=[text],
                model=self.model_name,
                task_type=self.task_type
            )
            return response[0]
        except Exception as e:
            st.error(f"Error embedding query: {e}")
            return []

def getpdftext(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        except Exception as e:
            try:
                doc = fitz.open(pdf)
                for page in doc:
                    text += page.get_text()
            except Exception as e2:
                st.error(f"Error processing PDF {pdf.name}: {e2}")
                return ""
    return text

def gettextchunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator=" ",  # Use space as separator
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

@st.cache_data
def process_pdf_and_chunk(pdf_docs):
    raw_text = getpdftext(pdf_docs)
    if not raw_text:
        return None

    text_chunks = gettextchunks(raw_text)

    if not text_chunks:
        st.error("No text chunks found. Check your PDFs and text extraction.")
        return None

    return text_chunks

def getvector(text_chunks):
    try:
        nomic_embeddings = NomicEmbeddings()
        vector_store = FAISS.from_texts(texts=text_chunks, embedding=nomic_embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

def inject_custom_css():
    st.markdown("""
        <style>
            /* ... (your CSS styles here) ... */
        </style>
    """, unsafe_allow_html=True)

def display_message(is_user: bool, content: str):
    message_type = "user" if is_user else "assistant"
    avatar_letter = "U" if is_user else "A"

    message_html = f"""
        <div class="chat-message {message_type}-message">
            <div class="avatar {message_type}-avatar">{avatar_letter}</div>
            <div class="message-content">{content}</div>
        </div>
    """
    st.markdown(message_html, unsafe_allow_html=True)

def get_conversation_chain(vector_store):  # Correctly sets up the chain
    llm = OpenAI(temperature=0)  # Or your preferred LLM
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),  # Use the vector store's retriever
        memory=memory,
    )
    return chain

def main():
    st.set_page_config(page_title="Question.io", page_icon=":shark:", layout="wide")
    inject_custom_css()

    st.title("Question.io")
    st.header("Ask questions and get AI Sourced Explanations for your Documents")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:
        st.title("Document Upload")
        pdf_docs = st.file_uploader("Upload your PDFs", type=["pdf"], accept_multiple_files=True)

        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                text_chunks = process_pdf_and_chunk(pdf_docs)
                if text_chunks:
                    st.success(f"Processed {len(text_chunks)} chunks successfully!")
                    st.session_state.text_chunks = text_chunks
                else:
                    st.error("Document processing failed.")

    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

    for message in st.session_state.chat_history:
        display_message(is_user=message["role"] == "user", content=message["content"])

    query = st.text_input("Ask a question about your documents:", key="query_input")

    if st.button("Send") and query:
        if "text_chunks" not in st.session_state:
            st.warning("Please upload and process documents first.")
            return

        st.session_state.chat_history.append({"role": "user", "content": query})
        display_message(is_user=True, content=query)

        with st.spinner("Thinking..."):
            vector_store = getvector(st.session_state.text_chunks)

            if vector_store:
                chain = get_conversation_chain(vector_store)  # Get the chain!

                try:
                    result = chain({"question": query})
                    answer = result['answer']

                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    display_message(is_user=False, content=answer)

                    st.session_state.query_input = ""
                except Exception as e:
                    st.error(f"Error generating response: {e}")
                    st.session_state.chat_history.append({"role": "assistant", "content": "I encountered an error. Please try again."})
                    display_message(is_user=False, content="I encountered an error. Please try again.")
            else:
                st.error("Could not process your question. Please try again.")

    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()