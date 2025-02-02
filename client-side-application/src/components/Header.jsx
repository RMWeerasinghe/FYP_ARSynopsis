import React,{useState} from "react";
import "../styles/Header.css";
import SideDrawer from "./Drawer";

const Header = ({downloadSummary}) => {

  
  const refreshSummary = () => {
    window.location.reload();
  }
  return (
    <div className="header">
        <div style={{display:"flex"}}>
            <SideDrawer/> 
            <h1 className="header-title" style={{marginTop : 2.5,marginLeft : 7}}>  PDF Summarizer</h1> 
        </div>
      
      <div className="header-buttons">
        <button className="header-button refresh-button" onClick={refreshSummary}><img width="20" height="20" src="https://img.icons8.com/ios-glyphs/30/refresh--v1.png" alt="refresh--v1"/>Refresh</button>
        <button className="header-button download-button" onClick={downloadSummary}><img width="20" height="20" src="https://img.icons8.com/forma-regular/24/download.png" alt="download"/>Download Summary</button>
      </div>
    </div>
  );
};

export default Header;
