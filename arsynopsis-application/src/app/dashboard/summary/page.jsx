import React from "react";
import Drawer from "@/components/drawer/drawer";

export const SummaryGenerator = () => {
    return (
        <div style={{flexDirection: 'column', display: 'flex'}}>
            <div>
                <Drawer />
            </div>
            <div>
                <h1>Summary Generator - For logged users</h1>
            </div>
            
        </div>
    )
}

export default SummaryGenerator;