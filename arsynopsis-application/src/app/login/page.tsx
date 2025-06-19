'use client';

import React, { useState } from "react";
import {loginUser} from "../../services/user-service";
import {useRouter} from "next/navigation";

const LoginPage = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);

  const router = useRouter();

  const handleClicked = () => {
        // Handle the click event for the "Sign up" link
        router.push("/signup");
    };

    const handleSubmit = async () => {
      // setError(null);
      setLoading(true);
      try {
        await loginUser(email, password);
        router.push("/"); 
      } catch (error) {
        console.error("Login error:", error);
        alert("Invalid email or password");
        // setError("Invalid email or password");
      }
      finally{
        setLoading(false);
      }
    }

  return (
    <div className="min-h-screen bg-gray-100 dark:bg-gray-900 flex items-center justify-center p-4">
      <div className="max-w-4xl w-full bg-white dark:bg-gray-800 rounded-xl shadow-lg overflow-hidden flex">
        {/* Left Panel - Login Form */}
        <div className="w-1/2 p-8">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-6 text-center">
            Sign In
          </h2>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Email
              </label>
              <input
                type="email"
                className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-500 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition-all"
                placeholder="your@email.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Password
              </label>
              <input
                type="password"
                className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-500 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition-all"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </div>

            <div className="flex items-center justify-between">
              <label className="flex items-center">
                <input
                  type="checkbox"
                  className="rounded border-gray-300 dark:border-gray-600 text-indigo-600 focus:ring-indigo-500"
                />
                <span className="ml-2 text-sm text-gray-600 dark:text-gray-400">
                  Remember me
                </span>
              </label>
              <a
                href="#"
                className="text-sm text-indigo-600 hover:text-indigo-500 dark:text-indigo-400 dark:hover:text-indigo-300"
              >
                Forgot password?
              </a>
            </div>

            <button
              type="submit"
              className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-medium py-2.5 rounded-lg transition-colors disabled:opacity-50"
              disabled={loading}
              onClick={(e) => {
                e.preventDefault();
                handleSubmit();
              }}
            >
              {loading ? "Signing In..." : "Sign In"}
            </button>
          </div>

          <div className="mt-6 text-center text-sm text-gray-600 dark:text-gray-400">
            Don't have an account?{" "}
            <a
              href="#"
              className="text-indigo-600 hover:text-indigo-500 dark:text-indigo-400 dark:hover:text-indigo-300 font-medium"
              onClick={handleClicked}
            >
              Sign up
            </a>
          </div>
        </div>

        {/* Right Panel - App Benefits */}
        <div className="w-1/2 bg-gradient-to-br from-indigo-600 to-purple-700 p-8 flex flex-col justify-center text-white">
          <div className="text-center">
            <h2 className="text-3xl font-bold mb-4">
              Hello, Friend!
            </h2>
            <p className="text-lg mb-8 opacity-90">
              Enter your personal details and start your journey with us
            </p>
            
            <div className="space-y-6 mb-8">
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-blue bg-opacity-20 rounded-full flex items-center justify-center">
                  <span className="text-sm font-bold">✓</span>
                </div>
                <span className="text-left">Access exclusive features and content</span>
              </div>
              
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-blue bg-opacity-20 rounded-full flex items-center justify-center">
                  <span className="text-sm font-bold">✓</span>
                </div>
                <span className="text-left">Personalized dashboard and analytics</span>
              </div>
              
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-blue bg-opacity-20 rounded-full flex items-center justify-center">
                  <span className="text-sm font-bold">✓</span>
                </div>
                <span className="text-left">Secure data storage and privacy</span>
              </div>
              
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-blue bg-opacity-20 rounded-full flex items-center justify-center">
                  <span className="text-sm font-bold">✓</span>
                </div>
                <span className="text-left">Sentiment analysis and Comparative analysis</span>
              </div>
            </div>

            <button
              className="border-2 border-white text-white px-8 py-2 rounded-full font-medium hover:bg-white hover:text-indigo-600 transition-colors"
              onClick={handleClicked}
            >
              SIGN UP
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LoginPage;
// import React from "react";
// import { useState } from "react";
// import {useRouter} from "next/navigation";
// import {loginUser} from "../../services/user-service";

// const LoginPage = () => {

//     const router = useRouter();

//     const [email, setEmail] = useState("");
//     const [password, setPassword] = useState("");
//     const [error, setError] = useState("");
//     const [loading, setLoading] = useState(false);


//     const handleClicked = () => {
//         // Handle the click event for the "Sign up" link
//         router.push("/signup");
//     };

//     const handleSubmit = async () => {
//       // setError(null);
//       setLoading(true);
//       try {
//         await loginUser(email, password);
//         router.push("/"); 
//       } catch (error) {
//         console.error("Login error:", error);
//         alert("Invalid email or password");
//         // setError("Invalid email or password");
//       }
//       finally{
//         setLoading(false);
//       }
//     }
//   return (
//     <div className="min-h-screen bg-gray-100 dark:bg-gray-900 flex items-center justify-center p-4">
//       <div className="max-w-md w-full bg-white dark:bg-gray-800 rounded-xl shadow-lg p-8">
//         <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-6 text-center">
//           Sign In
//         </h2>

//         <form className="space-y-4">
//           <div>
//             <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
//               Email
//             </label>
//             <input
//               type="email"
//               className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-500 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition-all"
//               placeholder="your@email.com"
//               onChange={(e) => setEmail(e.target.value)}
//             />
//           </div>

//           <div>
//             <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
//               Password
//             </label>
//             <input
//               type="password"
//               className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-500 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition-all"
//               placeholder="••••••••"
//               onChange={(e) => setPassword(e.target.value)}
//             />
//           </div>

//           <div className="flex items-center justify-between">
//             <label className="flex items-center">
//               <input
//                 type="checkbox"
//                 className="rounded border-gray-300 dark:border-gray-600 text-indigo-600 focus:ring-indigo-500"
//               />
//               <span className="ml-2 text-sm text-gray-600 dark:text-gray-400">
//                 Remember me
//               </span>
//             </label>
//             <a
//               href="#"
//               className="text-sm text-indigo-600 hover:text-indigo-500 dark:text-indigo-400 dark:hover:text-indigo-300"
//             >
//               Forgot password?
//             </a>
//           </div>

//           <button
//             type="submit"
//             className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-medium py-2.5 rounded-lg transition-colors"
          
//             onClick={(e) => {
//               e.preventDefault(); // Prevent default form submission
//               handleSubmit(); // Call the handleSubmit function
//             }}>
//             Sign In
//           </button>
//         </form>

//         <div className="mt-6 text-center text-sm text-gray-600 dark:text-gray-400">
//           Don't have an account?{" "}
//           <a
//             href="#"
//             className="text-indigo-600 hover:text-indigo-500 dark:text-indigo-400 dark:hover:text-indigo-300 font-medium"
//             onClick={handleClicked}
//           >
//             Sign up
//           </a>
//         </div>
//       </div>
//     </div>
//   );
// };

// export default LoginPage;
