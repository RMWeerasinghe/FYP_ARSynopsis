"use client";

import Image from "next/image";
import Navbar from "../components/navbar/navbar";
import UploadZone from "../components/upload-zone/upload-zone";
import { useState, useRef,useEffect } from "react";
import PdfViewer from "../components/pdf-viewer/pdf-viewer";
import SummaryDisplay from "../components/summary-display/summary-display";
import html2canvas from "html2canvas";
import { jsPDF } from "jspdf";
import data from "../data/data_new_format.json";
import "../components/pdf-viewer/pdf-viewer.css";
import './main.css'

export default function Home() {
  // const [pdfFile, setPdfFile] = useState(null);
  // const [summary, setSummary] = useState("");

  // const handleFileUpload = (file: any) => {
  //   setPdfFile(file);
  //   // Mock API call for summarization
  //   setTimeout(() => {
  //     setSummary("This is a summarized version of the uploaded PDF.");
  //   }, 2000);
  // };

  const [pdfFile, setPdfFile] = useState(null);
  const [summary, setSummary] = useState("");
  const [inputSentence, setInputSentence] = useState("");
  const [selectedSentence, setSelectedSentence] = useState([]);
  const [resultArray, setResultArray] = useState([]);
  const [pageNumber, setPageNumber] = useState();
  const [loadingState, setLoadingState] = useState(true);
  const [loadingSummary, setLoadingSummary] = useState(false);
  // const [divRef, setDivRef] = useState(null);

  const divRef = useRef(null);

  const handleFileUpload = (file: any) => {
    setPdfFile(file);
    console.log('File Name:', file.name);
    // Mock API call for summarization
    setTimeout(() => {
      setSummary("This is a summarized version of the uploaded PDF.");
    }, 2000);
  };

   useEffect(() => {
    if (pdfFile) {
      setLoadingSummary(true);
      setSummary(""); // clear summary before loading

      // Simulate loading delay (e.g., 3 seconds)
      const timer = setTimeout(() => {
        setSummary("This is a summarized version of the uploaded PDF.");
        setLoadingSummary(false);
      }, 15000);

      // Cleanup timeout if pdfFile changes or component unmounts
      return () => clearTimeout(timer);
    } else {
      setSummary("");
      setLoadingSummary(false);
    }
  }, [pdfFile]);

  const refreshSummary = () => {
    if (pdfFile) {
      // Simulate refreshing the summary
      setSummary("Refreshed summary of the uploaded PDF.");
    }
  };

  const downloadPdf = async () => {
    const element = divRef.current; // Get the reference to the div
    if (element) {
      const canvas = await html2canvas(element); // Convert the div to a canvas
      const imageData = canvas.toDataURL("image/png"); // Get the canvas as an image
      const pdf = new jsPDF("p", "mm", "a4"); // Create a PDF instance

      // Calculate the dimensions
      const imgWidth = 210; // A4 width in mm
      const imgHeight = (canvas.height * imgWidth) / canvas.width; // Maintain aspect ratio

      pdf.addImage(imageData, "PNG", 0, 0, imgWidth, imgHeight); // Add image to PDF
      pdf.save("download.pdf"); // Save the PDF
    }
  };

  return (
    <div>
      <Navbar />
      <div className="app">
      <UploadZone onFileUpload={handleFileUpload} />

      <div className="main-container">
        <div className="left-panel">
          {pdfFile ? (
            <PdfViewer
              file={pdfFile}
              inputSentenceArray={selectedSentence}
              setSentenceArrray={setResultArray}
              pageNumber={pageNumber}
              setLoadingState={setLoadingState}
            />
          ) : (
            <div
              className="pdf-container"
              style={{
                paddingLeft: 80,
                display: "flex",
                flexDirection: "row",
                alignItems: "center",
              }}
            >
              {/* <img src={NoFile} style={{width : 70 , height : 70 , filter: 'brightness(0) sepia(1) hue-rotate(90deg) saturate(500%)'}}/>  */}
              <img
                width="70"
                height="70"
                src="https://img.icons8.com/ios/50/737373/file--v1.png"
                alt="file--v1"
              />
              <p style={{ color: "#878a88", fontSize: 35, marginTop: 8 }}>
                No PDF File Available
              </p>
            </div>
          )}
          {/* <TestPV file={pdfFile} /> */}
        </div>
        <div className="right-panel">
          {pdfFile ? (
              loadingSummary ? (
                <div style={{ padding: 20, fontSize: 18, color: "#555" }}>
                  Loading summary...
                </div>
              ) : (
                <SummaryDisplay
                  summary={data.summary}
                  onSelectSentence={setSelectedSentence}
                  resultArray={resultArray}
                  changePageNumber={setPageNumber}
                  setDivRef={divRef}
                />
              )
            ) : (
              <div style={{ padding: 20, fontSize: 18, color: "#aaa" }}>
                Upload a PDF to see the summary here.
              </div>
            )}
        </div>
      </div>
      </div>
    </div>
  );
}
