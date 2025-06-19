"use client";

import Navbar from "../navbar/navbar";
import UploadZone from "../upload-zone/upload-zone";
import { useState, useRef, useEffect } from "react";
import PdfViewer from "../pdf-viewer/pdf-viewer";
import SummaryDisplay from "../summary-display/summary-display";
import html2canvas from "html2canvas";
import { jsPDF } from "jspdf";
import data from "../../data/data_new_format.json";
import "../pdf-viewer/pdf-viewer.css";
import "../../app/main.css";
import FileDetailCard from "../../components/file-detail-card/file-detail-card";
import "./summarizer.css";
import { addDocument } from "../../services/document-service";
import { getCurrentUser } from "@/services/user-service";

const Summerizer = () => {
  const [pdfFile, setPdfFile] = useState(null);
  const [summary, setSummary] = useState("");
  const [inputSentence, setInputSentence] = useState("");
  const [selectedSentence, setSelectedSentence] = useState([]);
  const [resultArray, setResultArray] = useState([]);
  const [pageNumber, setPageNumber] = useState();
  const [loadingState, setLoadingState] = useState(true);
  const [loadingSummary, setLoadingSummary] = useState(false);
  // const [divRef, setDivRef] = useState(null);

  // Document Details
  const [documentDetails, setDocumentDetails] = useState({
    company_name: "",
    category: "",
    doc_name: "",
    user_mail: "",
  });

  const divRef = useRef(null);

  const handleFileUpload = (file) => {
    setPdfFile(file);
    console.log("File Name:", file.name);
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

  useEffect(() => {
    if (!loadingState) {
      console.log(
        "Loading state changed to false, PDF file is ready for processing."
      );
    }
  }, [loadingState]);

  const formatFileSize = (sizeInBytes) => {
    const kb = sizeInBytes / 1024;
    if (kb < 1024) {
      // Less than 1 MB, show in KB with 2 decimals
      return `${kb.toFixed(2)} KB`;
    } else {
      const mb = kb / 1024;
      // Show in MB with 2 decimals
      return `${mb.toFixed(2)} MB`;
    }
  };

  const getPresignedUrl = async (filename, filetype,actiontype) => {
    try {
      const response = await fetch("http://localhost:8000/get-preassigned-url", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          filename,
          filetype,
          actiontype,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to get presigned URL");
      }

      const data = await response.json();
      return data.preassigned_url;
    } catch (error) {
      console.error("Error getting presigned URL:", error);
      throw error;
    }
  }

  const uploadObject = async (file) => {

    try {
      if (!file) {
        alert("Please select a file first!");
        return;
      }

      const preassigned_url = await getPresignedUrl(
        file.name,
        file.type,
        "put_object"
      );

      if (!preassigned_url) {
        alert("Invalid response from server");
        return;
      }

      const uploadResponse = await fetch(preassigned_url, {
        method: "PUT",
        headers: {
          "Content-Type": file.type,
        },
        body: file,
      });

      if (uploadResponse.ok) {
        alert("File uploaded successfully!");
      } else {
        const uploadErrorText = await uploadResponse.text(); // Get raw error text from S3
        alert(`Upload failed: ${uploadErrorText}`);
        console.error("S3 Upload Error:", uploadErrorText);
      }
    } catch (error) {
      console.error("Error during file upload:", error);
      alert("An error occurred while uploading the file. Please try again.");
    }

  }

  const UploadFile = async () => {
    try {
      if (!pdfFile) {
        alert("Please select a file first!");
        return;
      }

      
      await uploadObject(pdfFile);
      console.log("File uploaded successfully:", pdfFile.name);

      const summary_data = data;
      const summaryJsonString = JSON.stringify(summary_data, null, 2); // pretty print
      const summaryBlob = new Blob([summaryJsonString], { type: 'application/json' });
      
      const summaryFileName = `${pdfFile.name.split('.')[0]}_summary.json`;
      const summaryFile = new File([summaryBlob], summaryFileName, { type: '  application/json' });
      await uploadObject(summaryFile);

    } catch (error) {
      console.error("Error during file upload:", error);
      alert("An error occurred while uploading the file. Please try again.");
    }
  };

  const handleFileSave = async () => {
    const user_mail = await getCurrentUser();
    if (!user_mail) {
      console.error("No user is currently logged in.");
      return;
    }

    // const response = await UploadFile();
    if (!response) {
      console.error("File upload failed, cannot save document details.");
      return;
    }

    const documentData = {
      company_name: documentDetails.company_name,
      category: documentDetails.category,
      doc_name: documentDetails.doc_name,
      user_mail: user_mail,
    };

    console.log("Saving document details:", documentData);

    try {
      await addDocument(documentData);
      console.log("Document details saved successfully:", documentData);
      alert("Document details saved successfully!");
      document.getElementById("my_modal_1").close();
    } catch (error) {
      console.error("Error saving document details:", error);
    }
  };

  return (
    <div>
      {/* <Navbar /> */}
      <dialog id="my_modal_1" className="modal">
        <div className="modal-box max-w-md w-full">
          <h3 className="font-bold text-lg mb-6 text-center">
            Company Report Details
          </h3>

          <div className="flex flex-col space-y-4">
            <div>
              <label
                className="block text-sm font-medium mb-1"
                htmlFor="documentName"
              >
                Document Name
              </label>
              <input
                id="documentName"
                type="text"
                placeholder="Type here"
                className="input input-bordered w-full"
                onChange={(e) =>
                  setDocumentDetails({
                    ...documentDetails,
                    doc_name: e.target.value,
                  })
                }
              />
            </div>

            <div>
              <label
                className="block text-sm font-medium mb-1"
                htmlFor="companyName"
              >
                Company Name
              </label>
              <input
                id="companyName"
                type="text"
                placeholder="Type here"
                className="input input-bordered w-full"
                onChange={(e) =>
                  setDocumentDetails({
                    ...documentDetails,
                    company_name: e.target.value,
                  })
                }
              />
            </div>

            <div>
              <label
                className="block text-sm font-medium mb-1"
                htmlFor="companyCategory"
              >
                Company Category
              </label>
              <select
                id="companyCategory"
                defaultValue=""
                className="select select-bordered w-full"
                onChange={(e) =>
                  setDocumentDetails({
                    ...documentDetails,
                    category: e.target.value,
                  })
                }
              >
                <option value="" disabled>
                  Pick a Category...
                </option>
                <option>Banking</option>
                <option>Manufacturing</option>
                <option>Exports</option>
                <option>IT Industry</option>
                <option>Food</option>
                <option>Agriculture</option>
              </select>
            </div>
          </div>

          <div className="modal-action justify-end space-x-3 mt-6">
            <form method="dialog" className="flex space-x-3">
              <button
                className="btn btn-primary"
                type="submit"
                onClick={UploadFile}
              >
                Save
              </button>
              <button
                className="btn btn-outline"
                type="button"
                onClick={() => document.getElementById("my_modal_1").close()}
              >
                Close
              </button>
            </form>
          </div>
        </div>
      </dialog>

      <div className="app">
        {loadingState ? (
          <div style={{ width: "80%" }}>
            <UploadZone onFileUpload={handleFileUpload} />
          </div>
        ) : (
          <div
            className="file-card-row flex items-center gap-4"
            style={{
              marginLeft: 280,
              flexDirection: "row",
              backgroundColor: "#222a35",
              width: 480,
              paddingLeft: 20,
              paddingRight: 20,
              paddingTop: 10,
              paddingBottom: 10,
              borderRadius: 8,
            }}
          >
            <FileDetailCard
              fileName={pdfFile.name}
              fileSize={formatFileSize(pdfFile.size)}
              uploadDate={new Date().toLocaleDateString()}
            />
            <div className="flex items-center gap-3" style={{ marginLeft: 60 }}>
              <button
                className="file-button"
                onClick={() =>
                  document.getElementById("my_modal_1").showModal()
                }
                title="Save the Content"
              >
                <svg
                  className="size-[1.2em]"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={2}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M17 3H7a2 2 0 00-2 2v14a2 2 0 002 2h10a2 2 0 002-2V7l-4-4z"
                  />
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M16 21v-4a2 2 0 00-2-2H10a2 2 0 00-2 2v4"
                  />
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M12 11v6"
                  />
                  <circle
                    cx="12"
                    cy="13"
                    r="1"
                    stroke="currentColor"
                    strokeWidth={2}
                    fill="none"
                  />
                </svg>
              </button>

              <button
                onClick={() => {
                  setPdfFile(null);
                  setLoadingState(true);
                }}
                title="Cancel"
                className="file-button"
              >
                <svg
                  className="size-[1.2em]"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={2}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </button>
            </div>
          </div>
        )}

        <div className="main-container">
          <div className="left-panel" style={{ width: 600 }}>
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
};

export default Summerizer;
