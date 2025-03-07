import React, { useState } from 'react';
import axios from 'axios';
import { Button, Form } from 'react-bootstrap';

function Upload({ setProcessedFiles, setSessionId }) {
  const [files, setFiles] = useState([]);

  const handleFileChange = (e) => setFiles(e.target.files);

  const handleUpload = async () => {
    if (!files.length) {
      console.error("No files selected");
      return;
    }
    const formData = new FormData();
    for (let file of files) {
      formData.append('files', file);
    }
    console.log("Uploading files:", files); // Debug
    try {
      const response = await axios.post('http://localhost:5001/api/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      console.log("Upload response:", response.data); // Debug
      if (response.data && response.data.results) {
        setProcessedFiles(response.data.results.map(r => r.filename || r.status));
        setSessionId(response.data.session_id || '');
      } else {
        console.error("Invalid response format:", response.data);
      }
    } catch (error) {
      console.error('Upload failed:', error.message, error.response?.data); // Detailed error
    }
  };

  return (
    <div className="upload-section">
      <Form>
        <Form.Control type="file" multiple onChange={handleFileChange} accept=".pdf" />
        <Button onClick={handleUpload} variant="primary" className="mt-2">Process Documents</Button>
      </Form>
    </div>
  );
}

export default Upload;