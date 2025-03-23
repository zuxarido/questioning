import React, { useState, useEffect } from 'react';
import { Container, Row, Col } from 'react-bootstrap';
import Sidebar from './components/Sidebar';
import DocumentUpload from './components/DocumentUpload';
import ChatInterface from './components/ChatInterface';
import './styles.css';

function App() {
  const [sessionId, setSessionId] = useState(null);
  const [processedFiles, setProcessedFiles] = useState([]);
  const [chatHistory, setChatHistory] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);

  useEffect(() => {
    initializeSession();
  }, []);

  const initializeSession = async () => {
    try {
      const response = await fetch('http://127.0.0.1:5001/api/session');
      if (response.ok) {
        const data = await response.json();
        setSessionId(data.session_id);
        setChatHistory(data.chat_history);
        setProcessedFiles(data.processed_files);
      }
    } catch (error) {
      console.error('Failed to initialize session:', error);
    }
  };

  const clearSession = async () => {
    if (!processedFiles.length) return;
    
    try {
      const response = await fetch('http://127.0.0.1:5001/api/clear', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ session_id: sessionId }),
      });
      
      if (response.ok) {
        const data = await response.json();
        setSessionId(data.session_id);
        setChatHistory([]);
        setProcessedFiles([]);
      }
    } catch (error) {
      console.error('Failed to clear session:', error);
    }
  };

  return (
    <div className="app-container">
      <Container fluid>
        <Row>
          {/* Sidebar */}
          <Col md={3} className="sidebar">
            <Sidebar 
              sessionId={sessionId}
              processedFiles={processedFiles}
              onClearSession={clearSession}
            />
          </Col>
          
          {/* Main content */}
          <Col md={9} className="main-content">
            <h1 className="title">🤖 RAG Q&A Assistant</h1>
            <p className="subtitle">Upload documents and ask questions about their content. We only support Text PDF files for now.</p>
            
            <Row>
              {/* Document Upload */}
              <Col md={5}>
                <DocumentUpload 
                  sessionId={sessionId}
                  onFilesProcessed={(files) => setProcessedFiles(files)}
                  isProcessing={isProcessing}
                  setIsProcessing={setIsProcessing}
                />
              </Col>
              
              {/* Chat Interface */}
              <Col md={7}>
                <ChatInterface 
                  sessionId={sessionId}
                  chatHistory={chatHistory}
                  onChatUpdate={(history) => setChatHistory(history)}
                />
              </Col>
            </Row>
          </Col>
        </Row>
      </Container>
    </div>
  );
}

export default App;