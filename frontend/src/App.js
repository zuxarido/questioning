import React, { useState, useEffect } from 'react';
import { Container } from 'react-bootstrap';
import Sidebar from './components/Sidebar';
import Chat from './components/Chat';
import Upload from './components/Upload';
import './styles.css';

function App() {
  const [sessionId, setSessionId] = useState('');
  const [processedFiles, setProcessedFiles] = useState([]);
  const [chatHistory, setChatHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  
  // Load session from localStorage on component mount
  useEffect(() => {
    const savedSessionId = localStorage.getItem('ragSessionId');
    if (savedSessionId) {
      setSessionId(savedSessionId);
      fetchSessionInfo(savedSessionId);
    }
  }, []);
  
  const fetchSessionInfo = async (id) => {
    try {
      setIsLoading(true);
      const response = await fetch(`http://localhost:5001/api/session/${id}`);
      if (response.ok) {
        const data = await response.json();
        setProcessedFiles(data.processed_files || []);
      } else {
        localStorage.removeItem('ragSessionId');
        setSessionId('');
        setProcessedFiles([]);
      }
    } catch (error) {
      console.error('Error fetching session info:', error);
      localStorage.removeItem('ragSessionId');
      setSessionId('');
      setProcessedFiles([]);
    } finally {
      setIsLoading(false);
    }
  };
  
  const onClearSession = () => {
    setSessionId('');
    setProcessedFiles([]);
    setChatHistory([]);
    localStorage.removeItem('ragSessionId');
  };

  return (
    <Container fluid className="app-container p-0">
      <div className="d-flex h-100">
        <Sidebar 
          processedFiles={processedFiles}
          onClearSession={onClearSession}
          isLoading={isLoading}
          sessionId={sessionId}
          setSessionId={setSessionId}
          setProcessedFiles={setProcessedFiles}
          setIsLoading={setIsLoading}
        />
        <div className="flex-grow-1 d-flex flex-column">
          <Upload 
            sessionId={sessionId}
            setSessionId={setSessionId}
            setProcessedFiles={setProcessedFiles}
            setIsLoading={setIsLoading}
          />
          <Chat 
            sessionId={sessionId}
            chatHistory={chatHistory}
            setChatHistory={setChatHistory}
            isLoading={isLoading}
            setIsLoading={setIsLoading}
          />
        </div>
      </div>
    </Container>
  );
}

export default App;