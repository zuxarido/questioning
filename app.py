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



def main():
    st.set_page_config(page_title="Question.io", page_icon=":shark:", layout="wide")
    st.title("Question.io")
    st.header("Ask questions and get AI Sourced Explanations for your Documents")

    query = st.text_input("Enter your question here")

    with st.sidebar:
        st.title("Sidebar")
        st.write("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDFs", type=["pdf"], accept_multiple_files=True)

        if st.button("Submit Documents"):
            with st.spinner("Processing documents..."):
                text_chunks = process_pdf_and_chunk(pdf_docs)

                if text_chunks:
                    st.write(f"Number of text chunks: {len(text_chunks)}")
                    for i, chunk in enumerate(text_chunks):
                        st.write(f"Chunk {i+1} length: {len(chunk)}")
                    st.success("Documents processed successfully!")
                    st.session_state.text_chunks = text_chunks
                else:
                    st.error("Document processing failed.")

    if st.button("Ask Question") and query:
        if "text_chunks" in st.session_state and st.session_state.text_chunks:
            with st.spinner("Searching for answer..."):
                vector_store = getvector(st.session_state.text_chunks)

                if vector_store:
                    nomic_embeddings = NomicEmbeddings()
                    query_embedding = nomic_embeddings.embed_query(query)

                    docs_and_scores = vector_store.similarity_search_with_score(query_embedding)

                    if docs_and_scores:
                        highest_score = docs_and_scores[0][1]
                        SIMILARITY_THRESHOLD = 0.5  # Adjust as needed

                        context = ""
                        for doc, score in docs_and_scores:
                            context += doc.page_content + "\n\n"

                        prompt = f"""Context:
                        {context}

                        Question: {query}

                        Answer:"""

                        if highest_score < SIMILARITY_THRESHOLD:
                            st.warning(f"Your question is likely outside the scope of the provided documents. The highest similarity score is {highest_score:.2f}, which is below the threshold of {SIMILARITY_THRESHOLD}.")
                        else:
                            st.write(prompt) # print the prompt
                            # No LLM call in this version - you'll add that back in
                            st.write("Answer will go here (LLM not integrated in this version).")

                    else:
                        st.warning("No relevant information found in the documents for your question.")
                else:
                    st.error("Could not create vector store. Check document processing steps.")

        else:
            st.warning("Please upload and process documents first.")

    elif query:
        st.info("Please click 'Ask Question' to get results.")

if __name__ == '__main__':
    main()