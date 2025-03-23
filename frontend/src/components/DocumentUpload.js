import React, { useState } from 'react';
import { Card, Button, ProgressBar } from 'react-bootstrap';

function DocumentUpload({ sessionId, onFilesProcessed, isProcessing, setIsProcessing }) {
  const [uploadProgress, setUploadProgress] = useState(0);
  const [progressText, setProgressText] = useState('');

  const handleFileUpload = async (event) => {
    const files = event.target.files;
    if (!files.length) return;

    setIsProcessing(true);
    setUploadProgress(0);
    setProgressText('Preparing files for upload...');

    try {
      const formData = new FormData();
      formData.append('session_id', sessionId);

      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const progress = ((i + 1) / files.length) * 100;
        setUploadProgress(progress);
        setProgressText(`Processing ${file.name}...`);
        
        formData.append('files', file);
      }

      const response = await fetch('http://127.0.0.1:5001/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        
        if (data.results) {
          data.results.forEach(result => {
            if (result.status === 'no text extracted') {
              alert(`⚠️ Could not extract text from ${result.filename}. Please ensure it's a text-based PDF.`);
            } else if (result.status === 'processed') {
              console.log(`✅ Successfully processed ${result.filename}`);
            }
          });
        }

        if (!data.results || !data.results.length) {
          alert('⚠️ No text could be extracted from the uploaded files. We only support text-based PDF files for now.');
        }

        onFilesProcessed(data.processed_files);
      } else {
        throw new Error('Upload failed');
      }
    } catch (error) {
      console.error('Upload error:', error);
      alert('❌ Upload failed. Please check if the backend server is running.');
    } finally {
      setIsProcessing(false);
      setUploadProgress(0);
      setProgressText('');
    }
  };

  return (
    <Card className="upload-card">
      <Card.Header>
        <h2>📤 Upload Documents</h2>
      </Card.Header>
      <Card.Body>
        <div className="upload-container">
          <input
            type="file"
            accept=".pdf"
            multiple
            onChange={handleFileUpload}
            disabled={isProcessing}
            className="file-input"
            id="file-upload"
          />
          <label htmlFor="file-upload" className="file-label">
            {isProcessing ? 'Processing...' : 'Choose PDF files'}
          </label>
          
          {isProcessing && (
            <div className="progress-container">
              <ProgressBar 
                now={uploadProgress} 
                label={`${Math.round(uploadProgress)}%`}
                className="upload-progress"
              />
              <p className="progress-text">{progressText}</p>
            </div>
          )}
        </div>
      </Card.Body>
    </Card>
  );
}

export default DocumentUpload; 