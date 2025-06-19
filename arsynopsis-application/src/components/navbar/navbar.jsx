"use client";
// This is a client component
import React from "react";
import { useRouter } from "next/navigation";
import {
  logoutUser,
  onAuthStateChangedListener,
} from "../../services/user-service";
import { useEffect , useState } from "react";

const Navbar = () => {
  const router = useRouter();
  const [user, setUser] = useState(null);

  const handleClick = (path) => {
    router.push(`/${path}`); // Replace with your desired route
  };

  useEffect(() => {
    const unsubscribe = onAuthStateChangedListener((currentUser) => {
      setUser(currentUser);
    });
    return unsubscribe;
  }, []);

  const handleLogout = async () => {
    try {
      await logoutUser();
      router.push("/");
    } catch (error) {
      console.error("Logout error:", error);
    }
  };

  return (
    <div className="navbar bg-base-100 shadow-sm">
      <div className="flex-1">
        <a className="btn btn-ghost text-xl" onClick={() => router.push("/")}>
          ARSynopsis
        </a>
      </div>

      {user ? (
        <>
          <div className="flex-none hidden lg:block">
            <button className="btn" onClick={() => handleClick("dashboard")}>
              Dashboard
            </button>
          </div>
          <div className="flex-none hidden lg:block">
            <button className="btn" onClick={handleLogout}>
              Logout
            </button>
          </div>
        </>
      ) : (
        <>
          <div className="flex-none hidden lg:block">
            <button className="btn btn-neutral" onClick={() => handleClick("login")}>
              Login
            </button>
          </div>
          <div className="flex-none hidden lg:block" style={{ marginLeft: "10px" , marginRight: "10px" }}>
            <button className="btn btn-accent" onClick={() => handleClick("signup")}>
              SignUp
            </button>
          </div>
        </>
      )}

      {/* Cart and avatar dropdowns can remain unchanged */}
      
    </div>
  );
};

export default Navbar;
