import React from "react";
import { useDropzone } from "react-dropzone";
import "../styles/UploadZone.css";

const UploadZone = ({ onFileUpload }) => {
  const onDrop = (acceptedFiles) => {
    onFileUpload(acceptedFiles[0]);
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop });

  return (
    <div className="upload-zone" {...getRootProps()}>
      <input {...getInputProps()} />
      {isDragActive ? (
        <p>Drop the PDF here...</p>
      ) : (
        <div>
            <p>Drag & drop a PDF here, or click to select a file.</p>
            
        </div>
        
      )}
    </div>
  );
};

export default UploadZone;
