import {db} from '../services/firebase-config.js';
import { getDocs,addDoc,collection, query, where} from "firebase/firestore";




export async function addDocument(document) {
  try {
    console.log("Adding document to Firestore:", document);

    // Get a reference to the "document" collection
    const collectionRef = collection(db, "document");

    // Add a new document with auto-generated ID
    const docRef = await addDoc(collectionRef, {
      company_name: document.company_name,
      category: document.category,
      doc_name: document.doc_name,
      user_mail: document.user_mail,
    });

    console.log("Document written with ID: ", docRef.id);
  } catch (error) {
    console.error("Error adding document: ", error);
  }
};


/**
 * Retrieves documents from the "document" collection where user_mail matches the provided email.
 * @param {string} email - The email to filter documents by.
 * @returns {Promise<Array>} - Returns an array of matching document objects.
 */
export async function getDocumentsByEmail(email) {
  try {
    // Reference to the "document" collection
    const collectionRef = collection(db, "document");

    // Create a query against the collection where user_mail equals the provided email
    const q = query(collectionRef, where("user_mail", "==", email));

    // Execute the query
    const querySnapshot = await getDocs(q);

    // Map the results to an array of document data including the document ID
    const documents = querySnapshot.docs.map(doc => ({
      id: doc.id,
      ...doc.data()
    }));

    console.log(`Documents retrieved for email ${email}:`, documents);
    return documents;
  } catch (error) {
    console.error("Error retrieving documents by email: ", error);
    return [];
  }
}
