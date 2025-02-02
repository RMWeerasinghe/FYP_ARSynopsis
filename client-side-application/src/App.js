import React, { useState } from "react";
import UploadZone from "./components/UploadZone";
import PdfViewer from "./components/PdfViewer";
import SummaryDisplay from "./components/SummaryDisplay";
import Toolbar from "./components/Toolbar";
import "./styles/App.css";
import data from "./data/mock-data.json"
import TestPV from "./components/TestPdfViewer";
import Header from "./components/Header";
import html2canvas from "html2canvas";
import { jsPDF } from "jspdf";
import './styles/PdfViewer.css';
import NoFile from "./assets/prohibited.png"

const App = () => {
  const [pdfFile, setPdfFile] = useState(null);
  const [summary, setSummary] = useState("");
  const [inputSentence, setInputSentence] = useState("");
  const [selectedSentence, setSelectedSentence] = useState("");
  const [resultArray, setResultArray] = useState([]);
  const [pageNumber, setPageNumber] = useState();
  const [divRef, setDivRef] = useState(null);

  
  const handleFileUpload = (file) => {
    setPdfFile(file);
    // Mock API call for summarization
    setTimeout(() => {
      setSummary("This is a summarized version of the uploaded PDF.");
    }, 2000);
  };

  const refreshSummary = () => {
    if (pdfFile) {
      // Simulate refreshing the summary
      setSummary("Refreshed summary of the uploaded PDF.");
    }
  };

  const downloadPdf = async () => {
    const element = divRef.current; // Get the reference to the div
    const canvas = await html2canvas(element); // Convert the div to a canvas
    const imageData = canvas.toDataURL("image/png"); // Get the canvas as an image
    const pdf = new jsPDF("p", "mm", "a4"); // Create a PDF instance

    // Calculate the dimensions
    const imgWidth = 210; // A4 width in mm
    const imgHeight = (canvas.height * imgWidth) / canvas.width; // Maintain aspect ratio

    pdf.addImage(imageData, "PNG", 0, 0, imgWidth, imgHeight); // Add image to PDF
    pdf.save("download.pdf"); // Save the PDF
  };

  return (
    <div className="app">
      <Header downloadSummary={downloadPdf}/>
      <UploadZone onFileUpload={handleFileUpload} />
      {/* <input
        type="text"
        placeholder="Enter text to search"
        value={inputSentence}
        onChange={(e) => setInputSentence(e.target.value)}
      /> */}
      <div className="main-container">
        <div className="left-panel">
          {pdfFile ? 
            <PdfViewer 
              file={pdfFile}  
              inputSentence={selectedSentence} 
              setSentenceArrray = {setResultArray} 
              pageNumber = {pageNumber}/> 
            : 
            <div 
              className="pdf-container" 
              style={{paddingLeft:80,display : "flex",flexDirection : "row",alignItems : "center"}}>
               {/* <img src={NoFile} style={{width : 70 , height : 70 , filter: 'brightness(0) sepia(1) hue-rotate(90deg) saturate(500%)'}}/>  */}
               <img width="70" height="70" src="https://img.icons8.com/ios/50/737373/file--v1.png" alt="file--v1"/>
               <p style={{color : "#878a88" , fontSize : 35,marginTop : 8}}>No PDF File Available</p>
            </div>
            }
          {/* <TestPV file={pdfFile} /> */}
        </div>
        <div className="right-panel">
          <SummaryDisplay summary={data.summary_ril} onSelectSentence={setSelectedSentence} resultArray={resultArray} changePageNumber = {setPageNumber} setDivRef={setDivRef}/>
        </div>
      </div>
      {/* <Toolbar onRefresh={refreshSummary} onDownload={downloadSummary} /> */}
    </div>
  );
};

export default App;
