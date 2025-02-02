from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import sys
import os

from fastapi.middleware.cors import CORSMiddleware

# # Add the parent directory to sys.path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# # Import your custom summarization functions
# from utils.utils import display_document_with_clusters, get_condensed_report
# from Embeddings.preprocessing import get_all_sentences_array
# from Embeddings.embeddings import process_sentences_with_positional_encoding_updated
# from Clustering.clustering import k_means_cluster_document
# from Content_Selection.extractive_summarizer import pacsum_summarizer

sys.path.insert(0, 'C://Users//siriw//OneDrive//Desktop//New folder (2)//FYP_ARSynopsis//utils')
from sentence import Sentence
from utils import display_document_with_clusters, get_condensed_report

sys.path.insert(0, 'C://Users//siriw//OneDrive/Desktop//New folder (2)//FYP_ARSynopsis//Embeddings')
from preprocessing import get_all_sentences_array
from embeddings import process_sentences_with_positional_encoding_updated

sys.path.insert(0, 'C://Users//siriw//OneDrive/Desktop//New folder (2)//FYP_ARSynopsis//Clustering')
from clustering import k_means_cluster_document

sys.path.insert(0, 'C://Users//siriw//OneDrive/Desktop//New folder (2)//FYP_ARSynopsis//Content_Selection')
from extractive_summarizer import text_rank_summarizer, pacsum_summarizer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend URL: For now it has been set to allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/summarize")
async def summarize(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process the uploaded PDF
        sentence_array = get_all_sentences_array(temp_path)
        sentence_array = [sentence_tuple[0] for sentence_tuple in sentence_array]

        # Get embeddings
        document = process_sentences_with_positional_encoding_updated(sentence_array)

        # Perform clustering
        clustering, cluster_report = k_means_cluster_document(document, 120)

        # Generate summary using PacSum
        extracted_document = pacsum_summarizer(clustering, -2, 1, 0.6, 0.2)

        # Reorder the document
        condensed_report = get_condensed_report(extracted_document)

        print(condensed_report)

        # Combine sections into a single string
        summary = "\n\n".join(["\n".join(section) for section in condensed_report])

        # Remove temporary file
        os.remove(temp_path)

        return JSONResponse({"summary": condensed_report})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    
    

