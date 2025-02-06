import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
import os
from nomic import atlas

# Function to extract text from uploaded PDF files
def getpdftext(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split the extracted text into smaller chunks
def gettextchunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n ",  # Separator for splitting text
        chunk_size=1000,  # Maximum size of each chunk
        chunk_overlap=100,  # Overlapping characters between chunks
        length_function=len  # Function to determine text length
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

# Function to generate vector embeddings from text chunks
# I DO NOT LIKE THIS AT ALL NOT ONE BIT ABHI FIGURING OUT HOW TO INTEGRATE NOMIC API IN THIS YE MODEL LUND SE SLOW HAI ISS MODEL KI MAA KI CHUT MERE LAPTOP KI BHI MAA KI CHUT 
# 10 MB KI PDF KO BSDK EMBED KARNE MAI 15 MINUTE LAGE HAI
# testing out nomic integration and if it is working at a decent enough pace or not
def getvector(text):      
    try:
        # Load API key
        NOMIC_API_KEY = os.getenv('nk-nBZ4TcOKyJbQiD-AOYmvhxMgdPMvZq8Eu5ZodTUP9n8')
        if not NOMIC_API_KEY:
            raise ValueError("NOMIC_API_KEY not found in environment variables")
            
        # Create embeddings using Nomic
        embeddings = atlas.map_text(
            text=text,
            model_name="nomic-embed-text-v1",
            api_key=nk-nBZ4TcOKyJbQiD-AOYmvhxMgdPMvZq8Eu5ZodTUP9n8
        )
        
        # Create FAISS vector store
        vector_store = FAISS.from_texts(texts=text, embedding=embeddings)
        return vector_store
        
    except Exception as e:
        st.error(f"Error creating embeddings: {str(e)}")
        return None

# Main function to initialize the Streamlit app
def main():
    load_dotenv()  # Load environment variables
    
    st.set_page_config(page_title="Question.io", page_icon=":shark:", layout="wide")
    
    st.title("Question.io")
    st.header("Ask questions and get AI Sourced Explanations for your Documents")
    st.text_input("Enter your question here")  # Input field for user queries
    
    st.write("Choose the LLM model you want to use")
    st.selectbox("Select LLM Model", ["Llama", "DeepSeek"])  # Dropdown for model selection
    
    with st.sidebar:
        st.title("Sidebar")
        st.write("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDFs", type=["pdf"], accept_multiple_files=True)  # Upload PDFs
        
        if st.button("Submit"):
            with st.spinner("Processing"):
                raw_text = getpdftext(pdf_docs)  # Extract text
                text_chunks = gettextchunks(raw_text)  # Split text
                vector_store = getvector(text_chunks)  # Generate embeddings
                st.write(vector_store)  # Display vector store output
    
    st.button("Run")  # Placeholder for future execution
    
if __name__ == '__main__':
    main()
