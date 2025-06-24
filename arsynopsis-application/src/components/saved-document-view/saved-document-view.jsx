import React, { useEffect, useRef, useState } from "react";
import PdfViewer from "../pdf-viewer/pdf-viewer";
import SummaryDisplay from "../summary-display/summary-display";
import LoadingCircle from "../../assets/loading_circle.gif";
import Image from "next/image";

const SavedDocumentView = ({ onBackToLibrary, documentID, documenntName }) => {
  const [loading, setLoading] = React.useState(true);

  const [pdfFile, setPdfFile] = useState(null);
  const [jsonFile, setJsonFile] = useState(null);
  const [selectedSentence, setSelectedSentence] = useState([]);
  const [resultArray, setResultArray] = useState([]);
  const [pageNumber, setPageNumber] = useState();
  const [loadingState, setLoadingState] = useState(true);

  const divRef = useRef(null);

  const getPreAssignedURL = async (filename, filetype, directory) => {
    try {
      const response = await fetch(
        "http://localhost:8000/get-preassigned-url",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            filename: filename,
            filetype: filetype,
            directory: directory,
            actiontype: "get_object",
          }),
        }
      );

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      const data = await response.json();
      console.log("Response from pre-assigned URL:", data);
      console.log("Pre-assigned URL:", data.preassigned_url);
      return data.preassigned_url;
    } catch (error) {
      console.error("Error fetching pre-assigned URL:", error);
    }
  };

  const getObject = async (filename, filetype, directory) => {
    const preassignedURL = await getPreAssignedURL(
      filename,
      filetype,
      directory
    );

    console.log("Pre-assigned URL for getting:", preassignedURL);

    const response = await fetch(preassignedURL, {
      method: "GET",
    });

    console.log("Response status:", response.status);

    if (!response.ok) {
      throw new Error(`Failed to fetch file: ${response.statusText}`);
    }

    if (filetype === "application/json") {
      // For JSON, read as text and parse
      const text = await response.text();
      try {
        const json = JSON.parse(text);
        return json;
      } catch (error) {
        console.error("Error parsing JSON:", error);
        throw error;
      }
    } else if (filetype === "application/pdf") {
      // For PDF, return Blob or File
      const blob = await response.blob();
      // Optionally create a File object if needed:
      // const file = new File([blob], filename, { type: filetype });
      // return file;
      return blob;
    } else {
      // Handle other file types or throw error
      throw new Error(`Unsupported file type: ${filetype}`);
    }
  };

  useEffect(() => {
    const fetchData = async () => {
      const pdf_file = await getObject(
        documentID + "_document.pdf",
        "application/pdf",
        documentID
      );
      setPdfFile(pdf_file);
      const json_file = await getObject(
        documentID + "_summary.json",
        "application/json",
        documentID
      );
      setJsonFile(json_file);
    };

    fetchData();
  }, [documentID]);

  return (
    <div>
      <div
        className="breadcrumbs text-sm"
        style={{ marginLeft: 20, marginTop: 20 }}
      >
        <ul>
          <li>
            <a onClick={onBackToLibrary}>
              <svg
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                className="h-4 w-4 stroke-current"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"
                ></path>
              </svg>
              Library
            </a>
          </li>
          <li>{documenntName}</li>
        </ul>
      </div>

      {pdfFile && jsonFile ? (
        <div
          className="main-container"
          style={{ display: "flex", marginTop: 20, marginLeft: 20 }}
        >
          <div className="left-panel" style={{ width: 800 }}>
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
                  width: "100%",
                }}
              >
                {/* <img src={NoFile} style={{width : 70 , height : 70 , filter: 'brightness(0) sepia(1) hue-rotate(90deg) saturate(500%)'}}/>  */}
                <span
                  className="loading loading-infinity loading-xl"
                  style={{ width: 100, height: 100 }}
                ></span>
                <p style={{ color: "#878a88", fontSize: 35, marginTop: 8 }}>
                  No PDF File Available
                </p>
              </div>
            )}
            {/* <TestPV file={pdfFile} /> */}
          </div>
          <div className="right-panel">
            {jsonFile ? (
              <SummaryDisplay
                summary={jsonFile.summary}
                onSelectSentence={setSelectedSentence}
                resultArray={resultArray}
                changePageNumber={setPageNumber}
                setDivRef={divRef}
                displayRef={true}
              />
            ) : (
              <div style={{ padding: 20, fontSize: 18, color: "#aaa" }}>
                Upload a PDF to see the summary here.
              </div>
            )}
          </div>
        </div>
      ) : (
        <div
          style={{
            marginTop: 20,
            marginLeft: 20,
            width: 1200,
            height: 900,
            backgroundColor: "#1c232b",
            paddingTop: 250,
            paddingLeft: 500,
          }}
        >
          <span
            className="loading loading-bars loading-xl"
            style={{ width: 100, height: 100 }}
          ></span>
        </div>
      )}
    </div>
  );
};

export default SavedDocumentView;
