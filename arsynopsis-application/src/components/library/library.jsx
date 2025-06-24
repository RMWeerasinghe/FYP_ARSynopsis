"use client";
import React, { use, useEffect } from "react";
import Skeleton from "@/components/skeleton/skeleton";
import pdfImg from "../../assets/pdf.png";
import Image from "next/image";
import FileCard from "../file-card/file-card"
import FileCardDisplay from "../file-card-display/file-card-display"
import data from "../../data/company_list.json"
import {getDocumentsByEmail} from "../../services/document-service"
import { getCurrentUser } from "@/services/user-service";

const Library = ({onDocumentSelect}) => {

  const [documents, setDocuments] = React.useState([]);
  const [loading, setLoading] = React.useState(true);


  useEffect(() => {

    const fetchDocuments = async () => {
      try {
        const user_mail = await getCurrentUser();
        if (user_mail) {
          const email = user_mail;
          const docs = await getDocumentsByEmail(email);
          setDocuments(docs);
        } else {
          console.error("No user is currently logged in.");
        }
      } catch (error) {
        console.error("Error fetching documents:", error);
      }
    };

    fetchDocuments().then(() => {
      
      setLoading(false);
    }).catch((error) => {
      console.error("Error in fetchDocuments:", error);
      setLoading(false);
    });

    

  },[])
  

  return (
    <div>
      {! loading ? <ul
        className="list bg-base-100 rounded-box shadow-md"
        style={{
          width: 800,
          maxHeight: 700, // Set your desired height here
          overflowY: "auto", // Enable vertical scrolling
        }}
      >
        <li className="p-4 pb-2 text-xs opacity-60 tracking-wide">
          Saved Company Reports and Summaries
        </li>

        {documents.map((document, index) => (
          <FileCard
            key={index}
            company_name={document.company_name}
            doc_id = {document.doc_id}
            category={document.category}
            doc_name ={document.doc_name}
            onDocumentSelect={onDocumentSelect}
          />
        ))}

        {/* Placeholder for loading state */}  
        

        {/* <FileCard />
        <FileCard />
        <FileCard />
        <FileCard />
        <FileCard />
        <FileCard />
        <FileCard /> */}
      </ul> : <>Loading</>}
    </div>
  );
};

export default Library;
