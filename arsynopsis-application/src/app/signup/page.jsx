"use client";
import React from "react";
import { useRouter } from "next/navigation";
import Select from "react-select";
import makeAnimated from "react-select/animated";
import { useMemo , useState } from "react";
import countryList from 'react-select-country-list'
import selectStyles from "./styles"; // Assuming you have a separate file for styles
import {addUser} from "../../services/user-service.js";


const animatedComponents = makeAnimated();

const SignUp = () => {
  const router = useRouter();

  const options = useMemo(() => countryList().getData(), []);
  const [value, setValue] = useState('');
  const [userMail, setUserMail] = useState("");
  const [password, setPassword] = useState("");
  const [categories, setCategories] = useState([]);
  const [country, setCountry] = useState("");

  const handleCreateAccountClicked = async(e) => {
    const categoriesArray = categories.map(option => option.value);
    const user = {
      user_mail: userMail,
      password: password,
      categories: categoriesArray,
      region: country.label,
    };

    console.log("User data to be sent:", user);

    try {
      await addUser(user);
      console.log("User created successfully");
      router.push("/login");
    }
    catch (error) {
      console.error("Error creating user:", error);
      // Handle error appropriately, e.g., show a notification
    }
  }
    

  const changeHandler = value => {
    setValue(value)
  }

  const handleLoginClicked = () => {
    router.push("/login");
  };

  return (
    <div className="min-h-screen bg-gray-100 dark:bg-gray-900 flex items-center justify-center p-4">
      <div className="max-w-5xl w-full bg-white dark:bg-gray-800 rounded-xl shadow-lg overflow-hidden flex">
        {/* Left Panel - SignUp Form */}
        <div className="w-1/2 p-8">
          <div className="flex items-center justify-center mb-6">
            <img
              className="w-8 h-8 mr-2"
              src="https://flowbite.s3.amazonaws.com/blocks/marketing-ui/logo.svg"
              alt="logo"
            />
            <span className="text-2xl font-semibold text-gray-900 dark:text-white">
              ARSynopsis
            </span>
          </div>

          <h1 className="text-xl font-bold leading-tight tracking-tight text-gray-900 md:text-2xl dark:text-white mb-6 text-center">
            Create an account
          </h1>

          <div className="space-y-4 md:space-y-6">
            <div>
              <label
                htmlFor="email"
                className="block mb-2 text-sm font-medium text-gray-900 dark:text-white"
              >
                Your email
              </label>
              <input
                type="email"
                name="email"
                id="email"
                className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-primary-600 focus:border-primary-600 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
                placeholder="name@company.com"
                required
                onChange={(e) => {
                  setUserMail(e.target.value);
                }}
              />
            </div>
            <div>
              <label
                htmlFor="password"
                className="block mb-2 text-sm font-medium text-gray-900 dark:text-white"
              >
                Password
              </label>
              <input
                type="password"
                name="password"
                id="password"
                placeholder="••••••••"
                className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-primary-600 focus:border-primary-600 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
                required
                onChange={(e) => {
                  setPassword(e.target.value);
                }}
              />
            </div>
            <div>
              <label
                htmlFor="confirm-password"
                className="block mb-2 text-sm font-medium text-gray-900 dark:text-white"
              >
                Confirm password
              </label>
              <input
                type="password"
                name="confirm-password"
                id="confirm-password"
                placeholder="••••••••"
                className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-primary-600 focus:border-primary-600 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
                required
              />
            </div>
            <div>
              <label
                htmlFor="categories"
                className="block mb-2 text-sm font-medium text-gray-900 dark:text-white"
              >
                Categories
              </label>
              <Select
                closeMenuOnSelect={false}
                components={animatedComponents}
                isMulti
                options={[
                  { value: "technology", label: "Technology" },
                  { value: "health", label: "Health" },
                  { value: "finance", label: "Finance" },
                  { value: "education", label: "Education" },
                  { value: "entertainment", label: "Entertainment" },
                ]}
                className="basic-multi-select"
                classNamePrefix="select"
                styles={selectStyles}
                placeholder="Select categories"
                value={categories}
                onChange={(selectedOptions) => {
                  setCategories(selectedOptions);
                }}
              />
            </div>

            <div>
              <label
                htmlFor="region"
                className="block mb-2 text-sm font-medium text-gray-900 dark:text-white"
              >
                Region
              </label>
              <Select
               options={options}  
               styles={selectStyles}
               placeholder="Select country"
               value={country}
               onChange={(selectedOption) => {
                  setCountry(selectedOption);
                }}
              />
            </div>

            <div className="flex items-start">
              <div className="flex items-center h-5">
                <input
                  id="terms"
                  aria-describedby="terms"
                  type="checkbox"
                  className="w-4 h-4 border border-gray-300 rounded bg-gray-50 focus:ring-3 focus:ring-primary-300 dark:bg-gray-700 dark:border-gray-600 dark:focus:ring-primary-600 dark:ring-offset-gray-800"
                  required
                />
              </div>
              <div className="ml-3 text-sm">
                <label
                  htmlFor="terms"
                  className="font-light text-gray-500 dark:text-gray-300"
                >
                  I accept the{" "}
                  <a
                    className="font-medium text-primary-600 hover:underline dark:text-primary-500"
                    href="#"
                  >
                    Terms and Conditions
                  </a>
                </label>
              </div>
            </div>
            
            <button
              className="btn btn-info w-full text-white bg-primary-600 hover:bg-primary-700 focus:ring-4 focus:outline-none focus:ring-primary-300 font-medium rounded-lg text-sm px-5 py-2.5 text-center dark:bg-primary-600 dark:hover:bg-primary-700 dark:focus:ring-primary-800 bg-gradient-to-br from-indigo-600 to-purple-700"
              onClick={handleCreateAccountClicked}
            >
              Create an account
            </button>
            <p className="text-sm font-light text-gray-500 dark:text-gray-400 text-center">
              Already have an account?{" "}
              <a
                href="#"
                className="font-medium text-primary-600 hover:underline dark:text-primary-500"
                onClick={handleLoginClicked}
              >
                Login here
              </a>
            </p>
          </div>
        </div>

        {/* Right Panel - App Benefits */}
        <div className="w-1/2 bg-gradient-to-br from-indigo-600 to-purple-700 p-8 flex flex-col justify-center text-white">
          <div className="text-center">
            <h2 className="text-3xl font-bold mb-4">
              Welcome Back!
            </h2>
            <p className="text-lg mb-8 opacity-90">
              To keep connected with us please login with your personal info
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
              onClick={handleLoginClicked}
            >
              SIGN IN
            </button>
          </div>
        </div>
      </div>
      
    </div>
  );
};

export default SignUp;

// "use client";
// import React from "react";
// import { useRouter } from "next/navigation";
// import Select from "react-select";
// import makeAnimated from "react-select/animated";
// import { useMemo , useState } from "react";
// import countryList from 'react-select-country-list'
// import selectStyles from "./styles"; // Assuming you have a separate file for styles
// import {addUser} from "../../services/user-service.js";


// const animatedComponents = makeAnimated();

// const SignUp = () => {
//   const router = useRouter();

//   const options = useMemo(() => countryList().getData(), []);
//   const [value, setValue] = useState('');
//   const [userMail, setUserMail] = useState("");
//   const [password, setPassword] = useState("");
//   const [categories, setCategories] = useState([]);
//   const [country, setCountry] = useState("");

//   const handleCreateAccountClicked = async(e) => {
//     const categoriesArray = categories.map(option => option.value);
//     const user = {
//       user_mail: userMail,
//       password: password,
//       categories: categoriesArray,
//       region: country.label,
//     };

//     console.log("User data to be sent:", user);

//     try {
//       await addUser(user);
//       console.log("User created successfully");
//       router.push("/login");
//     }
//     catch (error) {
//       console.error("Error creating user:", error);
//       // Handle error appropriately, e.g., show a notification
//     }
//   }
    

//   const changeHandler = value => {
//     setValue(value)
//   }

//   const handleLoginClicked = () => {
//     router.push("/login");
//   };
//   return (
//     <section
//       className="bg-gray-50 dark:bg-gray-900"
//       style={{
//         flexDirection: "column",
//         alignItems: "center",
//         justifyContent: "center",
//       }}
//     >
//       <div className="flex flex-col items-center justify-center px-6 py-8 mx-auto md:h-screen lg:py-0">
//         <a
//           href="#"
//           className="flex items-center mb-6 text-2xl font-semibold text-gray-900 dark:text-white"
//         >
//           <img
//             className="w-8 h-8 mr-2"
//             src="https://flowbite.s3.amazonaws.com/blocks/marketing-ui/logo.svg"
//             alt="logo"
//           />
//           Synopto
//         </a>
//         <div className="w-full bg-white rounded-lg shadow dark:border md:mt-0 sm:max-w-md xl:p-0 dark:bg-gray-800 dark:border-gray-700">
//           <div className="p-6 space-y-4 md:space-y-6 sm:p-8">
//             <h1 className="text-xl font-bold leading-tight tracking-tight text-gray-900 md:text-2xl dark:text-white">
//               Create an account
//             </h1>
//             <div className="space-y-4 md:space-y-6" action="#">
//               <div>
//                 <label
//                   htmlFor="email"
//                   className="block mb-2 text-sm font-medium text-gray-900 dark:text-white"
//                 >
//                   Your email
//                 </label>
//                 <input
//                   type="email"
//                   name="email"
//                   id="email"
//                   className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-primary-600 focus:border-primary-600 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
//                   placeholder="name@company.com"
//                   required
//                   onChange={(e) => {
//                     setUserMail(e.target.value);
//                   }}
//                 />
//               </div>
//               <div>
//                 <label
//                   htmlFor="password"
//                   className="block mb-2 text-sm font-medium text-gray-900 dark:text-white"
//                 >
//                   Password
//                 </label>
//                 <input
//                   type="password"
//                   name="password"
//                   id="password"
//                   placeholder="••••••••"
//                   className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-primary-600 focus:border-primary-600 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
//                   required
//                   onChange={(e) => {
//                     setPassword(e.target.value);
//                   }}
//                 />
//               </div>
//               <div>
//                 <label
//                   htmlFor="confirm-password"
//                   className="block mb-2 text-sm font-medium text-gray-900 dark:text-white"
//                 >
//                   Confirm password
//                 </label>
//                 <input
//                   type="password"
//                   name="confirm-password"
//                   id="confirm-password"
//                   placeholder="••••••••"
//                   className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-primary-600 focus:border-primary-600 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
//                   required
//                 />
//               </div>
//               <div>
//                 <label
//                   htmlFor="confirm-password"
//                   className="block mb-2 text-sm font-medium text-gray-900 dark:text-white"
//                 >
//                   Categories
//                 </label>
//                 <Select
//                   closeMenuOnSelect={false}
//                   components={animatedComponents}
//                   isMulti
//                   options={[
//                     { value: "technology", label: "Technology" },
//                     { value: "health", label: "Health" },
//                     { value: "finance", label: "Finance" },
//                     { value: "education", label: "Education" },
//                     { value: "entertainment", label: "Entertainment" },
//                   ]}
//                   className="basic-multi-select"
//                   classNamePrefix="select"
//                   styles={selectStyles}
//                   placeholder="Select categories"
//                   value={categories}
//                   onChange={(selectedOptions) => {
//                     setCategories(selectedOptions);
//                   }}
//                 />
//               </div>

//               <div>
//                 <label
//                   htmlFor="confirm-password"
//                   className="block mb-2 text-sm font-medium text-gray-900 dark:text-white"
//                 >
//                   Region
//                 </label>
//                 <Select
//                  options={options}  
//                  styles={selectStyles}
//                  placeholder="Select country"
//                  value={country}
//                  onChange={(selectedOption) => {
//                     setCountry(selectedOption);
//                   }}
                
//                 />
//               </div>

//               <div className="flex items-start">
//                 <div className="flex items-center h-5">
//                   <input
//                     id="terms"
//                     aria-describedby="terms"
//                     type="checkbox"
//                     className="w-4 h-4 border border-gray-300 rounded bg-gray-50 focus:ring-3 focus:ring-primary-300 dark:bg-gray-700 dark:border-gray-600 dark:focus:ring-primary-600 dark:ring-offset-gray-800"
//                     required
//                   />
//                 </div>
//                 <div className="ml-3 text-sm">
//                   <label
//                     htmlFor="terms"
//                     className="font-light text-gray-500 dark:text-gray-300"
//                   >
//                     I accept the{" "}
//                     <a
//                       className="font-medium text-primary-600 hover:underline dark:text-primary-500"
//                       href="#"
//                     >
//                       Terms and Conditions
//                     </a>
//                   </label>
//                 </div>
//               </div>
              
//               <button
//                 // type="submit"
//                 className="w-full text-white bg-primary-600 hover:bg-primary-700 focus:ring-4 focus:outline-none focus:ring-primary-300 font-medium rounded-lg text-sm px-5 py-2.5 text-center dark:bg-primary-600 dark:hover:bg-primary-700 dark:focus:ring-primary-800"
//                 onClick={handleCreateAccountClicked}
//               >
//                 Create an account
//               </button>
//               <p className="text-sm font-light text-gray-500 dark:text-gray-400">
//                 Already have an account?{" "}
//                 <a
//                   href="#"
//                   className="font-medium text-primary-600 hover:underline dark:text-primary-500"
//                   onClick={handleLoginClicked}
//                 >
//                   Login here
//                 </a>
//               </p>
//             </div>
//           </div>
//         </div>
//       </div>
//       <div>
//         <h1>HII</h1>
//       </div>
//     </section>
//   );
// };

// export default SignUp;
