import React from "react";
import "../styles/Toolbar.css";

const Toolbar = ({ onRefresh, onDownload }) => (
  <div className="toolbar">
    <button onClick={onRefresh}>Refresh Summary</button>
    <button onClick={onDownload}>Download Summary</button>
  </div>
);

export default Toolbar;
