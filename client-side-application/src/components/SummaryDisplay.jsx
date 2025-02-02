import React, { useEffect, useRef } from "react";
import "../styles/SummaryDisplay.css";
import Carousel from "react-multi-carousel";
import "react-multi-carousel/lib/styles.css";
import html2canvas from "html2canvas";
import { jsPDF } from "jspdf";
import VerticallyCenteredModal from "./Model";

const SummaryDisplay = ({
  summary,
  onSelectSentence,
  resultArray,
  changePageNumber,
  setDivRef,
}) => {
  // const [resultArray, setResultArray] = useState([]);

  const divRef = useRef();
  const [modalShow, setModalShow] = React.useState(false);

  const handleSentenceClick = (sentence) => {
    onSelectSentence(sentence); // Send the selected sentence to App.js
  };

  const handleScroll = (page) => {
    changePageNumber(page);
  };

  useEffect(() => {
    setDivRef(divRef);
  }, [divRef]);

  // Split summary into sentences for highlighting
  const sentences = summary.split(/(?<=\.|\?|\!)\s/); // Split by sentence-ending punctuation

  return (
    <div>
      <VerticallyCenteredModal
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
      </VerticallyCenteredModal>
      
      <h3>Summary</h3>
      <div
        className="summary-display"
        ref={divRef}
        style={{ position: "relative" }}
      >
        <button onClick={() => setModalShow(true)} className="top-right-button">
        <img width="25" height="25" src="https://img.icons8.com/ios-filled/50/737373/expand--v1.png" alt="expand--v1"/>
        </button>
        {sentences.map((sentence, index) => (
          <span
            key={index}
            className="summary-sentence"
            onClick={() => handleSentenceClick(sentence)}
            style={{ cursor: "pointer", display: "inline", margin: "0 5px" , fontSize : '0.9rem' , fontWeight : 500}}
          >
            {sentence}
          </span>
        ))}
      </div>
      <h3>References</h3>
      <div className="reference-display">
        {resultArray?.length > 0 ? (
          <div>
            <ul>
            {resultArray.map((result, index) => (
              <li>
              <div
                key={index}
                style={{ marginLeft: "2px", marginRight: "40px" , fontSize : '0.9rem' , fontWeight : 400}}
              >
                <p>{result.current} &nbsp;
                  <a
                  href="#"
                  onClick={(e) => {
                    e.preventDefault(); // Prevent the default navigation behavior of the anchor tag
                    handleScroll(result.page); // Call the handleScroll function
                  }}
                >
                  Page {result.page - 2}
                </a></p>
                
              </div>
              </li>
            ))}
            </ul>
            
          </div>
        ) : (
          <><p style={{color : '#696b6a' , marginTop : 50,marginLeft : 80, fontSize : 20}}><img width="30" height="30" style = {{marginRight : 5}} src="https://img.icons8.com/ios/50/737373/circled-dot.png" alt="circled-dot"/> Select a summary section to view References.</p></>
        )}
        {/* {resultArray?.length > 0 ? (
        <div>
            <table>
                <thead>
                    <tr>
                        <th>Page</th>
                        <th>Similarity</th>
                        <th>Previous Sentence</th>
                        <th>Current Sentence</th>
                        <th>Next Sentence</th>
                        <th>Scroll</th>
                    </tr>
                </thead>
                <tbody>
                    {resultArray.map((result, index) => (
                        <tr key={index}>
                            <td>{result.page - 2}</td>
                            <td>{result.similarity}</td>
                            <td>{result.previous || "N/A"}</td>
                            <td>{result.current}</td>
                            <td>{result.next || "N/A"}</td>
                            <td>
                                <button onClick={() => {handleScroll(result.page)}}>Scroll to Page</button>
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
            
        </div>
        ) : (
          <></>
        )} */}
      </div>
    </div>
  );
};

export default SummaryDisplay;
