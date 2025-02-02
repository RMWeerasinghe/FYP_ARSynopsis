import React, { useState , useEffect , useRef} from "react";
import { Document, Page, pdfjs } from "react-pdf";
import "../styles/PdfViewer.css";
import { PDFHighlight } from "@pdf-highlight/react-pdf-highlight";
import 'react-pdf/dist/Page/AnnotationLayer.css';
import stringSimilarity from "string-similarity";
import axios from "axios";
import LoadingGIF from "../assets/pdf.gif";

// Set the workerSrc
pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

const PdfViewer = ({ file, inputSentence , setSentenceArrray,pageNumber}) => {
    const [numPages, setNumPages] = useState(null);
    const [sentencesByPage, setSentencesByPage] = useState([]); // Stores sentences for each page
    const [result, setResult] = useState(null); // Stores search 
    const [resultArray, setResultArray] = useState([]);
    const [response, setResponse] = useState(null);
    const [loading,setLoading] = useState(true);
    const [isLoaded, setIsLoaded] = useState(false);

    const pageRefs = useRef([]);
  
    const onDocumentLoadSuccess = ({ numPages }) => {
      setIsLoaded(true);
      setNumPages(numPages);
    };

    const scrollToPage = (pageNumber) => {
        if (pageRefs.current[pageNumber - 1]) {
          pageRefs.current[pageNumber - 1].scrollIntoView({ behavior: 'smooth' });
        }
      };
    
    useEffect(() => {
        scrollToPage(pageNumber);
    },[pageNumber]);
  
    // Convert file to ArrayBuffer
    const readFileAsArrayBuffer = (file) => {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsArrayBuffer(file);
      });
    };

    const processFile = async () => {
        console.log("Processing the PDF file")
        if (!file) {
            alert("No file provided.");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        console.log("Sending file to server...");
        console.time("Server response time");

        try {
            const res = await axios.post("http://localhost:8000/summarize", formData, {
                headers: {
                    "Content-Type": "multipart/form-data",
                },
            });
            setResponse(res.data);
            console.log("Response from server:", res.data);
            console.timeEnd("Server response time");
            setLoading(false)
        } catch (error) {
            console.error("Error processing file:", error);
        }
    };

    useEffect(() => {
        if (file) {
            processFile();
        }
    },[file]);
  
    // Extract text content from each page
    const extractTextFromPage = async (pdf, pageNumber) => {
      const page = await pdf.getPage(pageNumber);
      const textContent = await page.getTextContent();
      return textContent.items.map((item) => item.str).join(" "); // Combine text items
    };
  
    // Extract text for all pages and split into sentences
    // useEffect(() => {
    //   if (file) {
    //     console.log("Processing uploaded file...");
    //     readFileAsArrayBuffer(file).then((arrayBuffer) => {
    //       const loadingTask = pdfjs.getDocument({ data: arrayBuffer });
    //       loadingTask.promise.then(async (pdf) => {
    //         const allSentences = [];
    //         for (let i = 1; i <= pdf.numPages; i++) {
    //           const pageText = await extractTextFromPage(pdf, i);
    //           const sentences = pageText.split(
    //             /(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s/
    //           ); // Split into sentences
    //           allSentences.push(sentences); // Add sentences for this page

    //         }
    //         // console.log("Sentences extracted:", allSentences);
    //         setSentencesByPage(allSentences); // Update state with nested array
    //       });
    //     });
    //   }
    // }, [file]);
  
    // Search for the input sentence with approximate matching
    // useEffect(() => {
    //     setResultArray([]);
    //   if (inputSentence && sentencesByPage.length > 0) {
    //     console.log("Searching for input sentence...");
    //     for (let pageIndex = 0; pageIndex < sentencesByPage.length; pageIndex++) {
    //       const sentences = sentencesByPage[pageIndex];
  
    //       // Iterate through each sentence and calculate similarity
    //       for (let sentenceIndex = 0; sentenceIndex < sentences.length; sentenceIndex++) {
    //         const similarity = stringSimilarity.compareTwoStrings(
    //           inputSentence.toLowerCase(),
    //           sentences[sentenceIndex].toLowerCase()
    //         );
  
    //         if (similarity >= 0.8) {
    //           // Found an approximate match, extract surrounding sentences
    //           const surrounding = {
    //             page: pageIndex + 1,
    //             previous: sentences[sentenceIndex - 1] || null,
    //             current: sentences[sentenceIndex],
    //             next: sentences[sentenceIndex + 1] || null,
    //             similarity: (similarity * 100).toFixed(2) + "%", // Add similarity percentage
    //           };
    //           setResult(surrounding);
    //           setResultArray([...resultArray, surrounding]);
    //         //   return;
    //         }
    //       }
    //     }
    //     setResult(null); // Sentence not found
    //   }
    // }, [inputSentence, sentencesByPage]);

    useEffect(() => {
        setResultArray([]); // Clear previous results
        setSentenceArrray([]);
        if (inputSentence && response && response.sentences.length > 0) {
          console.log("Searching for input sentence...");
      
          response.sentences.forEach((pageData) => {
            const { page, sentences } = pageData; // Extract page number and sentences
      
            sentences.forEach((sentence, index) => {
              const inputLower = inputSentence.toLowerCase();
              const sentenceLower = sentence.toLowerCase();
      
              // Substring check
              const isSubstringMatch = sentenceLower.includes(inputLower);
      
              // Similarity check
              const similarity = stringSimilarity.compareTwoStrings(inputLower, sentenceLower);
      
              if (isSubstringMatch || similarity >= 0.7) {
                // Found an approximate match, extract surrounding sentences
                const surrounding = {
                  page, // Use the page number from backend data
                  previous: sentences[index - 1] || null,
                  current: sentence,
                  next: sentences[index + 1] || null,
                  similarity: (similarity * 100).toFixed(2) + "%", // Add similarity percentage
                };
      
                // Update the results
                setResultArray((prevResults) => [...prevResults, surrounding]);
                setSentenceArrray((prevResults) => [...prevResults, surrounding]);
                scrollToPage(page); // Scroll to the page with the result
              }
            });
          });
      
          setResult(null); // Clear single result since we are storing multiple results
        }
      }, [inputSentence, response]);
      
  
    return (
      <div className="pdf-container">
        {loading ? <><div style={{height : 100,padding : 249}}><img src={LoadingGIF} alt="My GIF" style={{ width: '100px', height: 'auto' }} /></div></>:<Document file={file} onLoadSuccess={onDocumentLoadSuccess}>
          <div className="pdf-scrollable">
            {Array.from(new Array(numPages), (el, index) => (
              <div key={`page_${index + 1}`} ref={(el) => (pageRefs.current[index] = el)}>
              <Page
                pageNumber={index + 1}
                renderTextLayer={false} // Disable text layer rendering
              />
            </div>
            ))}
          </div>
        </Document>}
        
        
        {/* {resultArray.length > 0 ? (
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
                                <button onClick={() => scrollToPage(result.page)}>Scroll to Page</button>
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
    );
  };
  
  export default PdfViewer;