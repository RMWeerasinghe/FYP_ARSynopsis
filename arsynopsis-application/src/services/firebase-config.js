// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
import { getFirestore } from "firebase/firestore";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyDHZiup1JJgQBeMmmloq4qicz5itJFwOCg",
  authDomain: "arsynopsis-fe056.firebaseapp.com",
  projectId: "arsynopsis-fe056",
  storageBucket: "arsynopsis-fe056.firebasestorage.app",
  messagingSenderId: "1079772833707",
  appId: "1:1079772833707:web:f5a3216044874ab61354ee",
  measurementId: "G-4F1H7NMG0J"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
// const analytics = getAnalytics(app);

// db initialization
export const db = getFirestore(app);