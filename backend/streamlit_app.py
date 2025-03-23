import streamlit as st
import requests
import json
import os
from pathlib import Path

# Configure the page with a modern theme
st.set_page_config(
    page_title="RAG Q&A Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 2rem;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1E88E5;
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Upload container */
    .uploadFile {
        border: 2px dashed #1E88E5;
        border-radius: 10px;
        padding: 2rem;
        margin: 1rem 0;
        background-color: rgba(30, 136, 229, 0.05);
    }
    
    /* Chat messages */
    .stChatMessage {
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        animation: fadeIn 0.3s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Buttons */
    .stButton button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Sidebar */
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    
    /* File list */
    .file-item {
        background-color: rgba(30, 136, 229, 0.1);
        padding: 0.5rem 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background-color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# Constants
BACKEND_URL = "http://127.0.0.1:5001"

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False

def initialize_session():
    """Initialize a new session with the backend"""
    with st.spinner("Initializing session..."):
        response = requests.get(f"{BACKEND_URL}/api/session")
        if response.status_code == 200:
            data = response.json()
            st.session_state.session_id = data['session_id']
            st.session_state.chat_history = data['chat_history']
            st.session_state.processed_files = data['processed_files']
            return True
        return False

def upload_files(files):
    """Upload files to the backend with progress tracking"""
    if not files:
        return
    
    try:
        files_data = []
        progress_text = "Preparing files for upload..."
        progress_bar = st.progress(0)
        
        for idx, file in enumerate(files):
            progress = (idx + 1) / len(files)
            progress_text = f"Processing {file.name}..."
            progress_bar.progress(progress, text=progress_text)
            
            # First check if file has content
            file_content = file.read()
            if len(file_content) == 0:
                st.error(f"❌ {file.name} appears to be empty")
                continue
                
            file.seek(0)
            files_data.append(('files', (file.name, file_content, 'application/pdf')))
        
        if not files_data:
            progress_bar.empty()
            return
            
        data = {'session_id': st.session_state.session_id}
        
        with st.spinner("Uploading to server..."):
            response = requests.post(f"{BACKEND_URL}/api/upload", files=files_data, data=data)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check if any files were actually processed
                if not data.get('results'):
                    st.warning("⚠️ No text could be extracted from the uploaded files. We only support text-based PDF files for now.")
                else:
                    # Check individual file results
                    for result in data['results']:
                        if result['status'] == 'no text extracted':
                            st.warning(f"⚠️ Could not extract text from {result['filename']}. Please ensure it's a text-based PDF.")
                        elif result['status'] == 'processed':
                            st.success(f"✅ Successfully processed {result['filename']}")
                
                st.session_state.processed_files = data['processed_files']
            else:
                st.error(f"❌ Upload failed: {response.text}")
        
        progress_bar.empty()
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.warning("Please check if the backend server is running.")

def send_query(question):
    """Send a query to the backend with improved error handling"""
    if not question:
        return
    
    data = {
        'question': question,
        'session_id': st.session_state.session_id
    }
    
    try:
        with st.spinner("🤔 Thinking..."):
            response = requests.post(f"{BACKEND_URL}/api/query", json=data)
            
            if response.status_code == 200:
                data = response.json()
                st.session_state.chat_history = data['chat_history']
                return data['answer']
            else:
                st.error("❌ Failed to get response")
                if response.text:
                    st.code(response.text)
                return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

def clear_session():
    """Clear the current session with confirmation"""
    if st.session_state.processed_files:
        if st.sidebar.button("⚠️ Confirm Clear", type="secondary"):
            with st.spinner("Clearing session..."):
                data = {'session_id': st.session_state.session_id}
                response = requests.post(f"{BACKEND_URL}/api/clear", json=data)
                
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.session_id = data['session_id']
                    st.session_state.chat_history = []
                    st.session_state.processed_files = []
                    st.success("🗑️ Session cleared!")
                    st.rerun()

# Main UI
st.title("🤖 RAG Q&A Assistant")
st.markdown("Upload documents and ask questions about their content. We only support Text PDF files for now.")

# Initialize session if not exists
if not st.session_state.session_id:
    if not initialize_session():
        st.error("❌ Failed to initialize session")
        st.stop()

# Sidebar
with st.sidebar:
    st.header("📊 Session Management")
    
    # Session info
    st.caption(f"Session ID: {st.session_state.session_id}")
    
    if st.session_state.processed_files:
        st.header("📚 Processed Files")
        for file in st.session_state.processed_files:
            st.markdown(f"""
                <div class="file-item">
                    📄 {file}
                </div>
            """, unsafe_allow_html=True)
        
        # Clear session button
        st.button("🗑️ Clear Session", type="primary", on_click=clear_session)
    else:
        st.info("📥 No files uploaded yet")

# Main content area
col1, col2 = st.columns([2, 3])

with col1:
    st.header("📤 Upload Documents")
    with st.container():
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="You can upload multiple PDF files"
        )

        if uploaded_files:
            num_files = len(uploaded_files)
            st.caption(f"Selected {num_files} file{'s' if num_files > 1 else ''}")
            
            if st.button("📤 Process Files", type="primary", disabled=st.session_state.is_processing):
                st.session_state.is_processing = True
                upload_files(uploaded_files)
                st.session_state.is_processing = False

with col2:
    st.header("💬 Chat Interface")
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message['role']):
                st.write(message['content'])
    
    # Query input
    if st.session_state.processed_files:
        if prompt := st.chat_input("Ask a question about your documents..."):
            with st.chat_message("user"):
                st.write(prompt)
            
            with st.chat_message("assistant"):
                response = send_query(prompt)
                if response:
                    st.write(response)
    else:
        st.info("👆 Please upload documents to start asking questions")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        Built with ❤️ using Streamlit and RAG
    </div>
    """, 
    unsafe_allow_html=True
) 