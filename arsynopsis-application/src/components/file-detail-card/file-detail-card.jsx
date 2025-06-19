import React from "react";
import pdfImg from "../../assets/pdf.png";
import Image from "next/image";

const FileDetailCard = ({ fileName, fileSize, uploadDate }) => {
  return (
    <div className="file-card-row flex items-center gap-4">
      <div>
        <Image src={pdfImg} alt="pdf" width={64} height={64} />
      </div>
      <div className="flex-1">
        <div>{fileName || "Untitled"}</div>
        <div className="text-xs uppercase font-semibold opacity-60">
          {fileSize || "Unknown"}
        </div>
      </div>
    </div>
  );
};

export default FileDetailCard;
