Document Q&A Assistant


A Retrieval-Augmented Generation (RAG) system for querying document content using natural language processing.

Overview
The Document Q&A Assistant is an advanced tool designed to process and query document content efficiently. Built with a Retrieval-Augmented Generation (RAG) architecture, it allows users to upload PDF and image files, extract text (including via OCR), and ask questions based on the document content. Leveraging state-of-the-art natural language processing (NLP) techniques, the system retrieves relevant information from the documents and provides precise answers through an intuitive web interface. This project demonstrates a practical application of AI in document management and information retrieval, suitable for educational, research, and professional use cases.

Objectives
Enable users to upload and process multiple document types (PDFs, PNGs, JPGs).
Extract text from documents, including scanned images, using client-side OCR.
Store document content in a vector database for efficient retrieval.
Provide a user-friendly interface for querying documents with natural language.
Ensure modularity and scalability through a well-structured codebase.
Features
Feature	Description
Document Upload	Supports PDFs and images (PNG, JPG) with multi-file upload capability.
Text Extraction	Extracts text from PDFs using PyMuPDF/PyPDF2 and images via Tesseract.js.
OCR Support	Client-side OCR for processing scanned documents and images.
Question Answering	Answers queries based on document content using Grok and Pinecone.
Interactive UI	Streamlit-based interface with chat history and session management.
Technology Stack
Programming Language: Python 3.x
NLP and AI:
Hugging Face Transformers (sentence-transformers/bert-base-nli-mean-tokens)
LangChain (text splitting, embeddings, retrieval)
ChatGroq (Grok model for question answering)
Vector Database: Pinecone
Document Processing: PyMuPDF, PyPDF2, Tesseract.js (client-side OCR)
Web Framework: Streamlit
Dependencies: Managed via requirements.txt
Environment: Configured with .env for API keys
Project Structure
text
Wrap
Copy
DocQnA/
├── app.py                   # Main application entry point
├── utils/                  # Utility modules
│   ├── __init__.py
│   ├── api_key_manager.py  # API key validation
│   ├── text_processor.py   # Text cleaning and chunking
│   ├── pdf_processor.py    # PDF text extraction
│   ├── vector_store.py     # Pinecone vector operations
│   ├── qa_chain.py         # QA chain logic
│   └── file_preprocessor.py # File processing with OCR
├── ui/                     # UI components
│   ├── __init__.py
│   └── streamlit_ui.py     # Streamlit interface
├── logs/                   # Log files
├── requirements.txt        # Project dependencies
└── .env                    # Environment variables (API keys)
Installation
Clone the Repository:
bash
Wrap
Copy
git clone https://github.com/jaskaransngh/DocQnA.git
cd DocQnA
Install Dependencies:
bash
Wrap
Copy
pip install -r requirements.txt
Configure Environment Variables:
Create a .env file in the root directory with the following:
text
Wrap
Copy
PINECONE_API_KEY=your-pinecone-key
PINECONE_ENVIRONMENT=your-pinecone-env
GROQ_API_KEY=your-grok-key
HUGGINGFACE_API_KEY=your-hf-key
Run the Application:
bash
Wrap
Copy
streamlit run app.py
Usage
Launch the App: Open your browser to http://localhost:8501 after running the command above.
Upload Documents: Use the sidebar to upload PDFs or images. Toggle "Enable OCR" for scanned files.
Process Files: Click "Process Documents" to extract text and store it in the vector database.
Ask Questions: Enter queries in the text box (e.g., "What’s on page 3?") and get answers based on the uploaded content.
View History: Chat history displays questions and responses for reference.
Implementation Details
Text Processing: Documents are split into 500-character chunks with 100-character overlap using RecursiveCharacterTextSplitter.
Embeddings: Text is embedded using BERT-based sentence transformers and stored in Pinecone.
Retrieval: Pinecone retrieves the top 4 relevant chunks for each query, filtered by session ID.
QA: Grok processes the retrieved context and query to generate answers.
OCR: Tesseract.js performs client-side text extraction for images and scanned PDFs, with progress feedback in the UI.
Limitations
OCR Performance: Client-side Tesseract.js may be slow for large or complex documents due to browser constraints.
Scalability: Limited by Streamlit’s single-threaded nature and session state management.
File Size: Large files may cause delays or browser memory issues during OCR processing.
Future Enhancements
Integrate server-side OCR (e.g., pytesseract) for faster processing of large documents.
Add live preview of text chunks during processing.
Implement caching for frequent queries and document chunks.
Support additional file formats (e.g., DOCX, TXT).
Acknowledgments
Developed by Jaskaran Singh as a final-year college project.
Built with assistance from xAI’s Grok 2 for design and optimization.
Utilizes open-source libraries from Hugging Face, LangChain, and Streamlit communities.
License
This project is licensed under the MIT License. See the LICENSE file for details.
