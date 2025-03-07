import React from 'react';
import axios from 'axios';
import { Button, ListGroup } from 'react-bootstrap';

function Sidebar({ processedFiles, setProcessedFiles, setSessionId }) {
  const handleClear = async () => {
    try {
      const response = await axios.post('http://localhost:5001/api/clear');
      setProcessedFiles([]);
      setSessionId(response.data.session_id);
    } catch (error) {
      console.error('Clear failed:', error);
    }
  };

  return (
    <div className="sidebar-content">
      <h3>Document Management</h3>
      <ListGroup>
        {processedFiles.map((file, idx) => (
          <ListGroup.Item key={idx}>{file}</ListGroup.Item>
        ))}
      </ListGroup>
      <Button variant="danger" onClick={handleClear} className="mt-3">Clear All Documents</Button>
    </div>
  );
}

export default Sidebar;