from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import sys
import os
import time
from transformers import pipeline
import json
import torch

""" Local LLM Run """


sys.path.insert(0, 'C://GitHub//FYP_ARSynopsis//Utils')
from sentence import Sentence
from utils import get_condensed_report,create_output

sys.path.insert(0, 'C://GitHub//FYP_ARSynopsis//Embeddings')
from preprocessing import get_all_sentences_array
from embeddings import process_sentences_with_positional_encoding_updated

sys.path.insert(0, 'C://GitHub//FYP_ARSynopsis//Clustering')
from clustering import k_means_cluster_document

sys.path.insert(0, 'C://GitHub//FYP_ARSynopsis//Content_Selection')
from extractive_summarizer import gusum_summarizer

sys.path.insert(0,'C://GitHub//FYP_ARSynopsis//Sentiment_Analysis')
from Sentiment import sentiment_analyser



# import sys
# sys.path.append(r"C://Users//siriw//OneDrive//Desktop//New folder//FYP_ARSynopsis//Content_Selection")
# from extractive_summarizer import gusum_summarizer

# Acsess Pvt Dataset and Model - load from secret
from dotenv import load_dotenv
from huggingface_hub import login
import os
load_dotenv()

token = os.getenv("HUGGING_FACE_TOKEN")

login(token=token)

app = FastAPI()

@app.post("/summarize")
async def summarize(file: UploadFile = File(...)):
    try:
        start = time.time()
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
        clustering, cluster_report = k_means_cluster_document(document, 50)

        # Generate summary using PacSum
        # extracted_document = pacsum_summarizer(clustering, -2, 1, 0.6, 0.2)
        extracted_document = gusum_summarizer(clustering,0.3)
        print("=====================Extraction Completed=============================")

        
        # Reorder the document
        condensed_report = get_condensed_report(extracted_document)
        print(type(condensed_report))


        # summarizer = pipeline("summarization", model="ARSynopsis/long-t5-base-govreport_80K_batch_2")
        # summarizer = pipeline("summarization", model="ARSynopsis/long-t5-base-govreport_10K_batch_1_16KToken",device = "cuda" if torch.cuda.is_available() else "cpu")
        summarizer = pipeline("summarization", model="ARSynopsis/T5_Full_FineTune_V0.1_80K",device = "cuda" if torch.cuda.is_available() else "cpu")
        # summarizer = pipeline("summarization", model="ARSynopsis/BART_Base_Full_FineTune_V0.1_83K")


        summaries = []
        count=0
        for ext_summary in condensed_report:
            count=count+1
            input_length = len(ext_summary.split())  # Approximate word count
            max_len = max(75, int(0.5 * input_length))  # 50% of input length, but at least 75 words
            
            summary = summarizer(ext_summary, max_length=max_len, min_length=75, do_sample=False)
            summaries.append(summary[0]["summary_text"])
            print("cluster No : ",count," complete..")

        # generated_summary = "//n".join(summaries)



        end = time.time()
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("Time" ,round((end-start),2))

        # generated_summary=create_output(condensed_report,summaries)

        sentiment_results, sentiment_propotions = sentiment_analyser(extracted_document)

        print("=====================Sentiment Analysis Completed=============================")

        generated_summary = create_output(condensed_report, summaries, sentiment_results, sentiment_propotions)

        # Convert SectionSummary objects into a JSON-friendly format
        json_summary = [
            {
                "section_id": section.section_id,
                "summary": section.summary,
                "mapping": section.mapping,  # Ensure this is a list of strings
                "postives":section.high_sentiments["pos"],
                "negatives":section.high_sentiments["neg"],
                "pos_p":section.sentiment_prop["pos"],
                "neg_p":section.sentiment_prop["neg"],
                "neu_p":section.sentiment_prop["neu"]

            }
            for section in generated_summary
        ]
# ================================================================================================================================
        # Define output and summary folder
        output_folder = "output"
        summary_folder = "summary_folder"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        if not os.path.exists(summary_folder):
            os.makedirs(summary_folder)

        # Get file name without extension
        model_name = "LOngT5-16K"

        pdf_name = file.filename.rsplit('.', 1)[0]  # Safer split for file extension
        safe_model_name = model_name.replace("/", "_")  # Replace invalid characters
        output_file = os.path.join(output_folder, f"{pdf_name}_{safe_model_name}_JSON.txt")
        summary_file = os.path.join(summary_folder, f"{pdf_name}_{safe_model_name}_summary.txt")

        print(f"Saving summary to: {output_file}")  # Debugging print

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({"summary": json_summary}, f, indent=4, ensure_ascii=False)
            print("JSON File saved successfully!")  # Confirmation print
        except Exception as e:
            print(f"Error saving file: {e}")

        # Extract only summaries as a single string
        summary_text = "//n//n".join([section["summary"] for section in json_summary])

        try:
            # Save full JSON output
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({"summary": json_summary}, f, indent=4, ensure_ascii=False)

            # Save only summaries as a plain text file
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(summary_text)

            print("Summary Files saved successfully!")  # Confirmation print
        except Exception as e:
            print(f"Error saving file: {e}")

#=======================================================================================================================
            
        return JSONResponse({"summary": json_summary})

    except Exception as e:
        print(e)
        return JSONResponse({"error": str(e)}, status_code=500)
