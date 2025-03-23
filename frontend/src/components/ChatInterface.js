import React, { useState, useRef, useEffect } from 'react';
import { Card, Form, Button } from 'react-bootstrap';

function ChatInterface({ sessionId, chatHistory, onChatUpdate }) {
  const [question, setQuestion] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [chatHistory]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim() || isLoading) return;

    setIsLoading(true);
    try {
      const response = await fetch('http://127.0.0.1:5001/api/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: question,
          session_id: sessionId
        }),
      });

      if (response.ok) {
        const data = await response.json();
        onChatUpdate(data.chat_history);
        setQuestion('');
      } else {
        throw new Error('Failed to get response');
      }
    } catch (error) {
      console.error('Query error:', error);
      alert('❌ Failed to get response. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Card className="chat-card">
      <Card.Header>
        <h2>💬 Chat Interface</h2>
      </Card.Header>
      <Card.Body className="chat-body">
        <div className="messages-container">
          {chatHistory.map((message, index) => (
            <div 
              key={index} 
              className={`message ${message.role === 'user' ? 'user-message' : 'assistant-message'}`}
            >
              <div className="message-content">
                {message.content}
              </div>
              {message.role === 'assistant' && message.sources && (
                <div className="sources">
                  <h4>Sources:</h4>
                  <ul>
                    {message.sources.map((source, idx) => (
                      <li key={idx}>{source}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        <Form onSubmit={handleSubmit} className="chat-form">
          <Form.Group className="d-flex gap-2">
            <Form.Control
              type="text"
              placeholder="Ask a question..."
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              disabled={isLoading}
            />
            <Button 
              type="submit" 
              disabled={isLoading || !question.trim()}
              className="send-button"
            >
              {isLoading ? '🤔 Thinking...' : 'Send'}
            </Button>
          </Form.Group>
        </Form>
      </Card.Body>
    </Card>
  );
}

export default ChatInterface; 