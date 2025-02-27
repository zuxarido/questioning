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


Backend Setup

Navigate to Backend:

cd /path/to/DocQnA-SaaS/backend


Create Virtual Environment:

python -m venv .venv
source .venv/bin/activate


Install Dependencies:
Use the provided requirements.txt:

bash '''
pip install -r requirements.txt
'''

Verify:

pip list | grep -E "streamlit|flask|pymupdf|langchain|pinecone|huggingface|torch|requests|Pillow|flask-cors"

Configure Environment:
Copy .env.example to .env:

cp .env.example .env
Edit .env with your keys:
PINECONE_API_KEY=your-pinecone-key
PINECONE_ENVIRONMENT=your-pinecone-env
GROQ_API_KEY=your-grok-key
HUGGINGFACE_API_KEY=your-hf-key

Run Backend:
Start Flask:
python app.py
Access: http://localhost:5001 (or 5000 if unchanged).



Frontend Setup
Navigate to Frontend:
cd /path/to/DocQnA-SaaS/frontend
Install Dependencies:

npm install
npm install axios react-bootstrap bootstrap react-scripts

Verify Structure:
Check files:

ls -R
public/ should have index.html, manifest.json.
src/ should have App.js, components/, index.js, styles.css.

Run Frontend:
Start React:
npm start
Access: http://localhost:3000.
Usage
Upload PDFs: Select files via the frontend upload section and click “Process Documents”.
Ask Questions: Type a question in the chat area and click “Send”.
Clear Session: Click “Clear All Documents” in the sidebar to reset.
Monitor: View processed files in the sidebar and chat history below.


Usage

Upload Documents: Use the sidebar to upload PDFs or images. Toggle "Enable OCR" for scanned files.
Process Files: Click "Process Documents" to extract text and store it in the vector database.
Ask Questions: Enter queries in the text box (e.g., "What’s on page 3?") and get answers based on the uploaded content.
View History: Chat history displays questions and responses for reference.
