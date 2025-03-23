import React from 'react';
import { Card, Button } from 'react-bootstrap';

function Sidebar({ sessionId, processedFiles, onClearSession }) {
  return (
    <div className="sidebar-container">
      <Card className="session-card">
        <Card.Header>
          <h2>📊 Session Management</h2>
        </Card.Header>
        <Card.Body>
          <div className="session-info">
            <p className="session-id">Session ID: {sessionId}</p>
          </div>

          {processedFiles.length > 0 ? (
            <>
              <h3>📚 Processed Files</h3>
              <div className="file-list">
                {processedFiles.map((file, index) => (
                  <div key={index} className="file-item">
                    📄 {file}
                  </div>
                ))}
              </div>
              <Button 
                variant="danger" 
                className="clear-button"
                onClick={onClearSession}
              >
                🗑️ Clear Session
              </Button>
            </>
          ) : (
            <div className="no-files">
              <p>📥 No files uploaded yet</p>
            </div>
          )}
        </Card.Body>
      </Card>
    </div>
  );
}

export default Sidebar;