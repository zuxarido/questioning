# Document Q&A Assistant


*A Retrieval-Augmented Generation (RAG) system for querying document content using natural language processing.*

---

## Overview
The **Document Q&A Assistant** is an advanced tool designed to process and query document content efficiently. Built with a Retrieval-Augmented Generation (RAG) architecture, it allows users to upload PDF and image files, extract text (including via OCR), and ask questions based on the document content. Leveraging state-of-the-art natural language processing (NLP) techniques, the system retrieves relevant information from the documents and provides precise answers through an intuitive web interface. This project demonstrates a practical application of AI in document management and information retrieval, suitable for educational, research, and professional use cases.

---

## Objectives
- Enable users to upload and process multiple document types (PDFs, PNGs, JPGs).  
- Extract text from documents, including scanned images, using client-side OCR.  
- Store document content in a vector database for efficient retrieval.  
- Provide a user-friendly interface for querying documents with natural language.  
- Ensure modularity and scalability through a well-structured codebase.  

---

## Features
| Feature                   | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| Document Upload           | Supports PDFs and images (PNG, JPG) with multi-file upload capability.     |
| Text Extraction           | Extracts text from PDFs using PyMuPDF/PyPDF2 and images via Tesseract.js.  |
| OCR Support               | Client-side OCR for processing scanned documents and images.               |
| Question Answering        | Answers queries based on document content using Grok and Pinecone.         |
| Interactive UI            | Streamlit-based interface with chat history and session management.        |

---

## Technology Stack
- **Programming Language**: Python 3.x  
- **NLP and AI**:  
  - Hugging Face Transformers 
  - LangChain (text splitting, embeddings, retrieval)  
  - ChatGroq (Grok model for question answering)  
- **Vector Database**: Pinecone  
- **Document Processing**: PyMuPDF, PyPDF2, Tesseract.js (client-side OCR)  
- **Web Framework**: Streamlit  
- **Dependencies**: Managed via `requirements.txt`  
- **Environment**: Configured with `.env` for API keys  

---

## Installation
Clone the Repository:  
   ```bash
   git clone https://github.com/jaskaransngh/DocQnA.git
   ```


Install Dependencies:

```bash
pip install -r requirements.txt      
```

Configure Environment Variables:
Create a .env file in the root directory with the following:
```
PINECONE_API_KEY=your-pinecone-key
PINECONE_ENVIRONMENT=your-pinecone-env
GROQ_API_KEY=your-grok-key
HUGGINGFACE_API_KEY=your-hf-key
```


Run the Application:
streamlit run app.py

Usage
Launch the App: Open your browser to ```http://localhost:8501``` after running the command above.
Upload Documents: Use the sidebar to upload PDFs or images. Toggle "Enable OCR" for scanned files.
Process Files: Click "Process Documents" to extract text and store it in the vector database.
Ask Questions: Enter queries in the text box (e.g., "What’s on page 3?") and get answers based on the uploaded content.
View History: Chat history displays questions and responses for reference.
