# RAG Q&A System

A Retrieval-Augmented Generation (RAG) system that allows users to upload documents and ask questions about their content.

## Features

- PDF document upload and processing
- Text extraction and chunking
- Vector storage using Pinecone
- Question answering using Groq's LLM
- Modern Streamlit UI
- Session management
- Support for multiple documents

## Prerequisites

- Python 3.9+
- Node.js (for React frontend)
- API keys for:
  - Pinecone
  - Groq
  - Hugging Face

## Project Structure

```
questioning/
├── backend/
│   ├── app.py              # Flask backend
│   ├── streamlit_app.py    # Streamlit frontend
│   ├── requirements.txt    # Python dependencies
│   └── .env               # Environment variables
└── frontend/              # React frontend (optional)
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd questioning
```

### 2. Set Up Python Environment

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the `backend` directory with your API keys:

```env
PINECONE_API_KEY=your-pinecone-key
PINECONE_ENVIRONMENT=your-pinecone-env
GROQ_API_KEY=your-groq-key
HUGGINGFACE_API_KEY=your-hf-key
```

### 4. Running the Application

You can run the application using either the Streamlit frontend (recommended) or the Flask backend with React frontend.

#### Option 1: Streamlit Frontend (Recommended)

1. Start the Flask backend:
```bash
cd backend
python app.py
```
The backend will run on http://localhost:5001

2. In a new terminal, start the Streamlit frontend:
```bash
cd backend
streamlit run streamlit_app.py
```
The Streamlit UI will be available at http://localhost:8502

#### Option 2: React Frontend (Optional)

1. Start the Flask backend:
```bash
cd backend
python app.py
```

2. In a new terminal, start the React frontend:
```bash
cd frontend
npm install
npm start
```
The React UI will be available at http://localhost:3000

## Usage

1. **Upload Documents**:
   - Click the upload area or drag and drop PDF files
   - Wait for the processing to complete
   - You'll see success/error messages for each file

2. **Ask Questions**:
   - Type your question in the chat input
   - The system will search through your documents and provide relevant answers
   - Chat history is maintained throughout the session

3. **Manage Sessions**:
   - Use the sidebar to view uploaded files
   - Clear the session to start fresh
   - Session ID is displayed in the sidebar

## Troubleshooting

1. **Port Already in Use**:
   - If port 5001 is in use, you can kill the process:
     ```bash
     lsof -i :5001  # Find the process
     kill -9 <PID>  # Kill the process
     ```

2. **API Key Issues**:
   - Ensure all required API keys are set in `.env`
   - Check the logs for specific error messages

3. **Document Processing**:
   - Only text-based PDFs are supported
   - Scanned PDFs or images are not supported yet
   - Minimum text length of 50 characters is required

## Development

- The Flask backend handles document processing and API interactions
- The Streamlit frontend provides a modern, responsive UI
- Logs are stored in the `logs` directory
- Debug mode is enabled by default

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
