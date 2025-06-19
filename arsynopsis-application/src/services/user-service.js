import {db} from '../services/firebase-config.js';
import { doc,setDoc} from "firebase/firestore";
import {getAuth, createUserWithEmailAndPassword, signInWithEmailAndPassword, signOut, onAuthStateChanged } from "firebase/auth";



export async function addUser(user) {
    const auth = getAuth();
    // Create user with email and password   
    try {
        console.log("Adding user to Firestore:", user);
        const userCredential = await createUserWithEmailAndPassword(auth, user.user_mail, user.password);
        const userId = userCredential.user.uid; // Get the user ID from Firebase Auth

        // Add user data to Firestore
        await setDoc(doc(db, "user", userId), {
            user_mail: user.user_mail,
            categories: user.categories,
            region: user.region,
        });
    } catch (error) {
        console.error("Error adding user: ", error);
        
    }
}

/**
 * Logs in a user with email and password using Firebase Authentication.
 * @param {string} email - User email
 * @param {string} password - User password
 * @returns {Promise} Resolves with userCredential on success, rejects with error on failure.
 */
export async function loginUser(email, password) {
  const auth = getAuth();
  try {
    const userCredential = await signInWithEmailAndPassword(auth, email, password);
    return userCredential;
  } catch (error) {
    throw error;
  }
}


/**
 * Logout the current user
 */
export async function logoutUser() {
  const auth = getAuth();
  try {
    await signOut(auth);
  } catch (error) {
    throw error;
  }
}

/**
 * Listen for auth state changes
 * @param {function} callback - function(user) called on auth state change
 * @returns unsubscribe function
 */
export function onAuthStateChangedListener(callback) {
  const auth = getAuth();
  return onAuthStateChanged(auth, callback);
}

export function getCurrentUser() {
  const auth = getAuth();
  const user =  auth.currentUser;
  return user ? user.email : null;
}