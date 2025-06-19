import React from "react";

const FileUploadButton = ({ onFileUpload }) => {
  const onDrop = (acceptedFile) => {
    onFileUpload(acceptedFile);
  };

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      onDrop(file);
    }
  };

  return (
    <div>
      {/* Generate a upload button when I click the upload button the file selecting winow will apear. when i click the upload button file input should trigger */}
      <button
        className="btn btn-info"
        onClick={() => document.getElementById("fileInput").click()}
        style={{
          width: 300,
          minHeight: 80,
          color: "white",
          fontWeight: 500,
          fontSize: 24,
          boxShadow: '0 8px 24px 0 rgba(0, 180, 216, 0.4)'
        }}
      >
        Select a PDF file
      </button>
      <input
        id="fileInput"
        type="file"
        accept=".pdf"
        style={{ display: "none" }}
        onChange={handleFileSelect}
      />
    </div>
  );
};

export default FileUploadButton;
