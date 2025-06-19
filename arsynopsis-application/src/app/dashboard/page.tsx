'use client';
// This is a client component
import React from "react";
import { useRouter , usePathname } from "next/navigation";
import { useState , useEffect } from "react";
import Drawer from "@/components/drawer/drawer";
import Library from "@/components/library/library";
import Summary from "@/components/summarizer/summarizer";
import { onAuthStateChangedListener } from "../../services/user-service";
import { logoutUser } from "../../services/user-service";

const Dashboard = () => {
    const [selectedItem, setSelectedItem] = useState('summary');
    const [loading, setLoading] = useState(true);
    const [user, setUser] = useState(null);
  
    const router = useRouter();
    // const pathname = usePathname();
  
    // Extract the last part of the path to determine selected item
    // const selectedItem = pathname?.split('/').pop() || 'summary';
  
    const handleClick = (path : string) => {
      router.push(`/dashboard/${path}`); // Replace with your desired route
    };

    const handleLogout = async () => {
    try {
      await logoutUser();
      router.push("/");
    } catch (error) {
      console.error("Logout error:", error);
    }
  };
  
    const renderContent = () => {
      switch (selectedItem) {
        case 'library':
          return <Library />;
        case 'summary':
          return <Summary />;
        case 'sentiment':
          return <div>Sentiment Page Under Construction</div>
        case 'compare':
          return <div>Compare Page Under Construction</div>
        default:
          return <Summary />;
      }
    }

    useEffect(() => {
    
    const unsubscribe = onAuthStateChangedListener((currentUser : any) => {
      if (!currentUser) {
        // Not logged in, redirect to login page
        router.replace("/");
      } else {
        setUser(currentUser);
      }
      setLoading(false);
    });

    return () => unsubscribe();
  }, [router]);

  if (loading) {
    return (
      <div className="flex justify-center items-center min-h-screen">
        <p>Loading...</p>
      </div>
    );
  }

  if (!user) {
    // Redirecting, or you can return null here
    return null;
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
          ></label>
          <ul className="menu bg-base-200 text-base-content min-h-full w-80 p-4">
            {/* Sidebar content here */}
            <li style={{marginLeft: 'auto', marginRight: 'auto'}}>
              <img
                  src="https://avatars.githubusercontent.com/u/12345678?v=4"
                  alt="User Avatar"
                  className="w-20 h-16 rounded-full mb-2"
                />
            </li>
            <li style={{marginLeft: 'auto', marginRight: 'auto'}}>
                <label style={{marginLeft : 40}}>
                <span className="text-lg font-bold">Kamal</span>
              </label>
              <label>kamal@gmail.com</label>
              
              
            </li>
            <br />
            <hr style={{color : '#2a2f38'}}/>
            <li style={{marginLeft: 'auto', marginRight: 'auto',width: '100%',marginTop : 40}}>
              {/* <button className="btn" onClick={() => {setSelectedItem('library')}}>Library</button> */}
              <a onClick={() => {setSelectedItem('library')}} style={{paddingLeft : 110,height: 50,paddingTop : 15}}>Library</a>
            </li>
            <li style={{marginLeft: 'auto', marginRight: 'auto',width: '100%'}}>
            {/* <button className="btn" onClick={() => {setSelectedItem('summary')}}>Summary</button> */}
            <a onClick={() => {setSelectedItem('summary')}} style={{paddingLeft : 110,paddingTop : 15,height: 50}}>Summary</a>
            </li>

            <li style={{marginLeft: 'auto', marginRight: 'auto',width: '100%'}}>
            {/* <button className="btn" onClick={() => {setSelectedItem('summary')}}>Summary</button> */}
            <a onClick={() => {setSelectedItem('sentiment')}} style={{paddingLeft : 110,paddingTop : 15,height: 50}}>Sentiment</a>
            </li>

            <li style={{marginLeft: 'auto', marginRight: 'auto',width: '100%'}}>
            {/* <button className="btn" onClick={() => {setSelectedItem('summary')}}>Summary</button> */}
            <a onClick={() => {setSelectedItem('compare')}} style={{paddingLeft : 110,paddingTop : 15,height: 50}}>Compare</a>
            </li>

            <li style={{marginTop : 220}}>
              <button className="btn btn-neutral" onClick={handleLogout}>Logout</button>
            </li>

          </ul>
        </div>
      </div>
    );
};

export default Dashboard;
