import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import  HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS



def getpdftext(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def gettextchunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="/n ",
        chunk_size = 1000,
        chunk_overlap = 100,
        length_function = len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def getvector(text):      
    try:
        embeddings = HuggingFaceInstructEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cpu"}
           
        )
        vector_store = FAISS.from_texts(texts=text, embedding=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating embeddings: {str(e)}")
        return None


    
def main():
   
    load_dotenv()
    
    st.set_page_config(page_title="Question.io", page_icon=":shark:", layout="wide")

    st.title("Question.io")
    st.header("Ask questions and get Ai Sourced Explanations for your Documents")
    st.text_input("Enter your question here")
   
    st.write("Choose the LLM model you want to use")
    st.selectbox("Select LLM Model", ["Llama", "DeepSeek"])

    with st.sidebar:
        st.title("Sidebar")
        st.write("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDFs", type=["pdf"], accept_multiple_files=True)
        if st.button("Submit"):
            with st.spinner("Processing"):
                raw_text = getpdftext(pdf_docs)
                

                text_chunks = gettextchunks(raw_text)
                


                vector_store = getvector(text_chunks)
                st.write(vector_store)








    
    st.button("Run")

  


if __name__ == '__main__':
    main()