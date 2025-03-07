import React, { useState } from 'react';
import { Container, Row, Col } from 'react-bootstrap';
import Upload from './components/Upload';
import Chat from './components/Chat';
import Sidebar from './components/Sidebar';
import './styles.css';

function App() {
  const [chatHistory, setChatHistory] = useState([]);
  const [processedFiles, setProcessedFiles] = useState([]);
  const [sessionId, setSessionId] = useState('');

  return (
    <Container fluid className="app">
      <Row>
        <Col md={3} className="sidebar">
          <Sidebar processedFiles={processedFiles} setProcessedFiles={setProcessedFiles} setSessionId={setSessionId} />
        </Col>
        <Col md={9} className="main-content">
          <h1>Document Q&A Assistant</h1>
          <Upload setProcessedFiles={setProcessedFiles} setSessionId={setSessionId} />
          <Chat chatHistory={chatHistory} setChatHistory={setChatHistory} sessionId={sessionId} />
        </Col>
      </Row>
    </Container>
  );
}

export default App;