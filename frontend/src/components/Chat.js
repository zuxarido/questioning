import React, { useState, useRef, useEffect } from 'react';
import { Form, Alert, Spinner } from 'react-bootstrap';

const Chat = ({ chatHistory, setChatHistory, sessionId, isLoading, setIsLoading, processedFiles }) => {
  const [question, setQuestion] = useState('');
  const [error, setError] = useState('');
  const chatEndRef = useRef(null);

  // Scroll to bottom of chat when chat history updates
  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [chatHistory]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim()) {
      setError('Please enter a question');
      return;
    }
    if (!sessionId) {
      setError('Please upload documents first');
      return;
    }
    
    setError('');
    setIsLoading(true);
    
    // Add user message immediately for better UX
    const userMessage = { role: 'user', content: question };
    setChatHistory([...chatHistory, userMessage]);
    setQuestion('');
    
    try {
      const response = await fetch('http://localhost:5001/api/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: userMessage.content,
          session_id: sessionId
        }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to get response');
      }
      
      const data = await response.json();
      setChatHistory(data.chat_history || []);
    } catch (error) {
      console.error('Error querying API:', error);
      setError(`Error: ${error.message}`);
      // Remove the user message if there was an error
      setChatHistory(chatHistory);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-container">
      {error && <Alert variant="danger">{error}</Alert>}
      
      <div className="chat-messages">
        {chatHistory.length === 0 ? (
          <div className="empty-chat">
            <svg width="30" height="30" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <circle cx="12" cy="12" r="10" stroke="var(--accent-purple)" strokeWidth="2"/>
              <path d="M12 7V12" stroke="var(--accent-purple)" strokeWidth="2" strokeLinecap="round"/>
              <path d="M12 12L15 15" stroke="var(--accent-purple)" strokeWidth="2" strokeLinecap="round"/>
            </svg>
            <p>Hello! Please submit a document to start our conversation.</p>
          </div>
        ) : (
          chatHistory.map((msg, index) => (
            <div
              key={index}
              className={`message ${msg.role === 'user' ? 'user-message' : 'assistant-message'}`}
            >
              <div className="message-content">
                <p>{msg.content}</p>
              </div>
            </div>
          ))
        )}
        <div ref={chatEndRef} />
      </div>
      
      <Form onSubmit={handleSubmit} className="chat-form">
        <Form.Control
          type="text"
          placeholder={sessionId ? "Ask a question about your documents..." : "Please submit a document first..."}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          disabled={isLoading || !sessionId}
          className="chat-input"
        />
        <button
          type="submit"
          disabled={isLoading || !sessionId || !question.trim()}
          className="send-button"
        >
          {isLoading ? (
            <Spinner animation="border" size="sm" />
          ) : (
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M22 2L11 13" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M22 2L15 22L11 13L2 9L22 2Z" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          )}
        </button>
      </Form>
    </div>
  );
};

export default Chat;