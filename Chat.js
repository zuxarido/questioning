import React, { useState } from 'react';
import axios from 'axios';
import { Form, Button } from 'react-bootstrap';

function Chat({ chatHistory, setChatHistory, sessionId }) {
  const [question, setQuestion] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question || !sessionId) return;
    try {
      const response = await axios.post('http://localhost:5001/api/query', { question });
      setChatHistory(response.data.chat_history);
      setQuestion('');
    } catch (error) {
      console.error('Query failed:', error);
    }
  };

  return (
    <div className="chat-section">
      <div className="chat-history">
        {chatHistory.map(([role, msg], idx) => (
          <div key={idx} className={role === 'human' ? 'user-msg' : 'ai-msg'}>
            {msg}
          </div>
        ))}
      </div>
      <Form onSubmit={handleSubmit}>
        <Form.Control type="text" value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="Ask a question..." />
        <Button type="submit" variant="primary" className="mt-2">Send</Button>
      </Form>
    </div>
  );
}

export default Chat;