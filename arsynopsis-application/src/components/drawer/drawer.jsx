'use client';
// This is a client component
import React from "react";
import { useRouter , usePathname } from "next/navigation";
import { useState } from "react";


const Drawer = () => {
//   const [selectedItem, setSelectedItem] = useState('summary');

  const router = useRouter();
  const pathname = usePathname();

  // Extract the last part of the path to determine selected item
  const selectedItem = pathname?.split('/').pop() || 'summary';

  const handleClick = (path) => {
    router.push(`/dashboard/${path}`); // Replace with your desired route
  };

  

  const renderContent = () => {
    switch (selectedItem) {
      case 'library':
        return <h1>Library</h1>;
      case 'summary':
        return <h1>Summary</h1>;
      default:
        return <h1>Summary</h1>;
    }
  }

  return (
    <div className="drawer lg:drawer-open">
      <input id="my-drawer-2" type="checkbox" className="drawer-toggle" />
      <div className="drawer-content flex flex-col items-center justify-center">
        {renderContent()}
        <label
          htmlFor="my-drawer-2"
          className="btn btn-primary drawer-button lg:hidden"
        >
          Open drawer
        </label>
      </div>
      <div className="drawer-side">
        <label
          htmlFor="my-drawer-2"
          aria-label="close sidebar"
          className="drawer-overlay"
        >
          Synopto
        </label>
        <ul className="menu bg-base-200 text-base-content min-h-full w-80 p-4">
          {/* Sidebar content here */}
          <li>
            <label className="label">
              <span className="label-text text-2xl font-bold">Synopto</span>
            </label>
          </li>
          <li>
            <button className="btn" onClick={() => {handleClick('library')}}>Library</button>
          </li>
          <li>
          <button className="btn" onClick={() => {handleClick('summary')}}>Summary</button>
          </li>
        </ul>
      </div>
    </div>
  );
};

export default Drawer;
