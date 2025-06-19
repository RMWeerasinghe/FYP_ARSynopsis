"use client";
import React, { useEffect, useRef, useState } from "react";
import "./summary-display.css";
// import Carousel from "react-multi-carousel";
import "react-multi-carousel/lib/styles.css";
// import html2canvas from "html2canvas";
// import { jsPDF } from "jspdf";
import VerticallyCenteredModal from "../model/vertically-centered-modal";

const SummaryDisplay = ({
  summary,
  onSelectSentence,
  resultArray,
  changePageNumber,
  setDivRef,
}) => {
  // const [resultArray, setResultArray] = useState([]);

  //   const divRef = useRef();
  const [modalShow, setModalShow] = React.useState(false);

  const [selectedId, setSelectedId] = useState(null);

  const handleSentenceClick = (mapping, section_id) => {
    setSelectedId(section_id);
    onSelectSentence(mapping); // Send the selected sentence to App.js
  };

  const handleScroll = (page) => {
    changePageNumber(page);
  };

  //   useEffect(() => {
  //     setDivRef(divRef);
  //   }, [divRef]);

  // Split summary into sentences for highlighting
  // const sentences = summary.split(/(?<=\.|\?|\!)\s/); // Split by sentence-ending punctuation
  // const sentences_v2 = null;

  return (
    <div>
      {/* <VerticallyCenteredModal
        show={modalShow}
        onHide={() => setModalShow(false)}
      >
        {sentences.map((sentence, index) => (
          <span
            key={index}
            className="summary-sentence"
            onClick={() => handleSentenceClick(sentence)}
            style={{ cursor: "pointer", display: "inline", margin: "0 5px" }}
          >
            {sentence}
          </span>
        ))}
      </VerticallyCenteredModal> */}

      <h3>Summary</h3>
      <div
        className="summary-display"
        ref={setDivRef}
        style={{ position: "relative" }}
      >
        <button onClick={() => setModalShow(true)} className="top-right-button">
          <img
            width="25"
            height="25"
            src="https://img.icons8.com/ios-filled/50/737373/expand--v1.png"
            alt="expand--v1"
          />
        </button>
        {
          /* {sentences.map((sentence, index) => (
          <span
            key={index}
            className="summary-sentence"
            onClick={() => handleSentenceClick(sentence)}
            style={{ cursor: "pointer", display: "inline", margin: "0 5px" , fontSize : '0.9rem' , fontWeight : 500}}
          >
            {sentence}
          </span>
        ))} */
          summary.map(({ section_id, summary, mapping }) => {
            const isSelected = selectedId === section_id;

            return (
              <span
                key={section_id}
                className="summary-sentence"
                onClick={() => handleSentenceClick(mapping, section_id)}
                style={{
                  cursor: "pointer",
                  display: "inline",
                  margin: "0 5px",
                  fontSize: "0.9rem",
                  fontWeight: 500,
                  color: "#fff",
                  backgroundColor: isSelected ? "#4a557a" : "transparent", // Highlight color when selected
                  borderRadius: "4px",
                  padding: isSelected ? "2px 6px" : "0",
                  transition: "background-color 0.3s ease",
                }}
                onMouseEnter={(e) => {
                  if (!isSelected)
                    e.currentTarget.style.backgroundColor = "#3a4461";
                }}
                onMouseLeave={(e) => {
                  if (!isSelected)
                    e.currentTarget.style.backgroundColor = "transparent";
                }}
              >
                {summary}
              </span>
            );
          })
        }
      </div>
      <h3>References</h3>
      <div className="reference-display">
        {resultArray?.length > 0 ? (
          <div>
            <ul>
              {resultArray.map((result, index) => (
                <li>
                  {/* <div
                    key={index}
                    style={{
                      marginLeft: "2px",
                      marginRight: "40px",
                      fontSize: "0.9rem",
                      fontWeight: 400,
                    }}
                  >
                    <p>
                      {result.current} &nbsp;
                      <a
                        href="#"
                        onClick={(e) => {
                          e.preventDefault(); // Prevent the default navigation behavior of the anchor tag
                          handleScroll(result.page); // Call the handleScroll function
                        }}
                      >
                        Page {result.page - 2}
                      </a>
                    </p>
                  </div> */}
                  <div className="collapse bg-base-100 border border-base-300">
                    <input type="radio" name="my-accordion-1" defaultChecked />
                    <div className="collapse-title font-semibold">
                      Page {result.page - 2}
                    </div>
                    <div className="collapse-content text-sm">
                      {result.current}
                      <a
                        href="#"
                        onClick={(e) => {
                          e.preventDefault(); // Prevent the default navigation behavior of the anchor tag
                          handleScroll(result.page); // Call the handleScroll function
                        }}
                        style={{
                          marginLeft: "10px",
                          color: "#4a5568",
                          textDecoration: "underline",
                        }}
                        
                      >
                        Navigate
                      </a>
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        ) : (
          <>
            <p
              style={{
                color: "#696b6a",
                marginTop: 50,
                marginLeft: 80,
                fontSize: 20,
              }}
            >
              <img
                width="30"
                height="30"
                style={{ marginRight: 5 }}
                src="https://img.icons8.com/ios/50/737373/circled-dot.png"
                alt="circled-dot"
              />{" "}
              Select a summary section to view References.
            </p>
          </>
        )}
      </div>
    </div>
  );
};

export default SummaryDisplay;
