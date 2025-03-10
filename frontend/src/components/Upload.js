import React, { useState, useRef, useEffect } from 'react';
import { Form, Alert, ProgressBar } from 'react-bootstrap';

const Upload = ({ sessionId, setSessionId, setProcessedFiles, setIsLoading }) => {
  const [files, setFiles] = useState([]);
  const [fileError, setFileError] = useState('');
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef(null);
  const progressIntervalRef = useRef(null);

  // Cleanup on component unmount
  useEffect(() => {
    return () => {
      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current);
      }
    };
  }, []);

  const handleFileChange = (e) => {
    const selectedFiles = Array.from(e.target.files || []);
    
    if (selectedFiles.length === 0) {
      setFileError('No files selected');
      return;
    }

    // Validate files
    const invalidFiles = selectedFiles.filter(file => {
      const fileName = file.name.toLowerCase();
      return !fileName.endsWith('.pdf') || file.type !== 'application/pdf';
    });

    if (invalidFiles.length > 0) {
      setFileError(`Only PDF files are supported. Invalid files: ${invalidFiles.map(f => f.name).join(', ')}`);
      return;
    }
    
    setFileError('');
    setFiles(selectedFiles);
    handleUpload(selectedFiles);
  };

  const handleUpload = async (selectedFiles) => {
    if (!selectedFiles?.length) {
      setFileError('Please select at least one PDF file to upload.');
      return;
    }
    
    setUploading(true);
    setIsLoading(true);
    setUploadProgress(0);
    
    const formData = new FormData();
    selectedFiles.forEach(file => {
      formData.append('files', file);
    });
    
    if (sessionId) {
      formData.append('session_id', sessionId);
    }
    
    try {
      // Simulate progress
      progressIntervalRef.current = setInterval(() => {
        setUploadProgress(prev => {
          const newProgress = prev + 5;
          return newProgress >= 90 ? 90 : newProgress;
        });
      }, 500);
      
      const response = await fetch('http://localhost:5001/api/upload', {
        method: 'POST',
        body: formData,
      });
      
      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current);
      }
      setUploadProgress(100);
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Upload failed with status: ${response.status} - ${errorText}`);
      }
      
      const data = await response.json();
      setSessionId(data.session_id);
      setProcessedFiles(data.processed_files || []);
      localStorage.setItem('ragSessionId', data.session_id);
      
      // Reset form
      setFiles([]);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    } catch (error) {
      console.error('Upload error:', error);
      setFileError(`Error uploading files: ${error.message}`);
    } finally {
      setUploading(false);
      setIsLoading(false);
      setTimeout(() => {
        if (!uploading) {
          setUploadProgress(0);
        }
      }, 1000);
    }
  };

  const triggerFileInput = () => {
    if (fileInputRef.current && !uploading) {
      fileInputRef.current.click();
    }
  };

  return (
    <div className="upload-container">
      <div className="upload-area" onClick={triggerFileInput}>
        <div className="upload-icon">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 5V19" stroke="#a176ff" strokeWidth="2" strokeLinecap="round"/>
            <path d="M5 12L12 5L19 12" stroke="#a176ff" strokeWidth="2" strokeLinecap="round"/>
          </svg>
        </div>
        <div className="upload-text">
          <p><strong>Upload documents</strong></p>
          <small>Multiple files supported</small>
        </div>
        <Form.Control
          type="file"
          multiple
          accept=".pdf"
          onChange={handleFileChange}
          ref={fileInputRef}
          disabled={uploading}
          style={{ display: 'none' }}
        />
      </div>
      
      {fileError && <Alert variant="danger">{fileError}</Alert>}
      
      {uploadProgress > 0 && (
        <ProgressBar 
          animated 
          now={uploadProgress} 
          label={`${uploadProgress}%`} 
          className="mb-3" 
          variant="primary"
        />
      )}
    </div>
  );
};

export default Upload;