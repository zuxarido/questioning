import React, { useState } from 'react';
import { Button, ListGroup, Badge, Form } from 'react-bootstrap';
import Upload from './Upload';

const Sidebar = ({ processedFiles, onClearSession, isLoading, sessionId, setSessionId, setProcessedFiles, setIsLoading }) => {
  const [enableOCR, setEnableOCR] = useState(false);

  return (
    <div className="sidebar">
      <h3>Documents</h3>
      
      <div className="ocr-switch">
        <Form.Check 
          type="switch"
          id="ocr-switch"
          label="Enable OCR"
          checked={enableOCR}
          onChange={() => setEnableOCR(!enableOCR)}
        />
        <div className="ocr-text">
          Extract text from images and scanned documents
        </div>
      </div>
      
      <Upload 
        sessionId={sessionId} 
        setSessionId={setSessionId} 
        setProcessedFiles={setProcessedFiles} 
        setIsLoading={setIsLoading} 
      />
      
      {processedFiles.length > 0 && (
        <>
          <ListGroup className="document-list">
            {processedFiles.map((filename, index) => (
              <ListGroup.Item key={index} className="document-item d-flex justify-content-between align-items-center">
                {filename}
                <Badge bg="primary" pill className="document-badge">PDF</Badge>
              </ListGroup.Item>
            ))}
          </ListGroup>
          
          <Button
            variant="outline-danger"
            onClick={onClearSession}
            disabled={isLoading || processedFiles.length === 0}
            className="mt-3 w-100"
          >
            Clear All Documents
          </Button>
        </>
      )}
    </div>
  );
};

export default Sidebar;