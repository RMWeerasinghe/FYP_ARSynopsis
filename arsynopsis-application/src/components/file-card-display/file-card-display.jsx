import React from "react";
import pdfImg from "../../assets/pdf.png";
import Image from "next/image";
import FileCard from "../file-card/file-card";

const FileCardDisplay = () => {
  return (
    <div>
      <ul
        className="list bg-base-100 rounded-box shadow-md"
        style={{ width: 800 }}
      >
        <li className="p-4 pb-2 text-xs opacity-60 tracking-wide">
          Most played songs this week
        </li>

        <FileCard />
        <FileCard />
        <FileCard />
        <FileCard />
        <FileCard />
        <FileCard />
        <FileCard />

        
      </ul>
    </div>
  );
};

export default FileCardDisplay;
