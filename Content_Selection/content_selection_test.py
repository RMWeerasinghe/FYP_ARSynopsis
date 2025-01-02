"""
Test script to test a long report dataset using the segmentation + extractive summarization framework.
"""
from datasets import Dataset, load_metric
from extractive_summarizer import text_rank_summarizer, pacsum_summarizer
import pandas as pd
import sys

sys.path.insert(0, 'C://GitHub//FYP_ARSynopsis//utils')
from sentence import Sentence
from utils import display_document_with_clusters, sort_by_longest_seq

sys.path.insert(0, 'C://GitHub//FYP_ARSynopsis//Embeddings')
from preprocessing import get_all_sentences_array
from embeddings import process_sentences_with_positional_encoding_updated

sys.path.insert(0, 'C://GitHub//FYP_ARSynopsis//Clustering')
from clustering import k_means_cluster_document

sys.path.insert(0, 'C://GitHub//FYP_ARSynopsis//Content_Selection')
from extractive_summarizer import pacsum_summarizer

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

import time

def extraction_pipe(text_or_pdf_path:str, is_pdf:bool = False)->str:
    try:
        start = time.time()
        if is_pdf:
            sentence_array = get_all_sentences_array(text_or_pdf_path)
            sentence_array = [sentence_tuple[0] for sentence_tuple in sentence_array]
        else:
            sentence_array = nltk.tokenize.sent_tokenize(text_or_pdf_path)

        # Get embeddings with positional encoding
        document = process_sentences_with_positional_encoding_updated(sentence_array)

        k = min(100,len(document)//2)

        # K-Means Clustering with dimensionality reduction 
        clustering, cluster_report = k_means_cluster_document(document,k)

        # extractive summarization using Text Rank
        #extracted_document = text_rank_summarizer(clustering, 0.3)

        # extractive summarization using PacSum
        extracted_document = pacsum_summarizer(clustering,-2,1,0.6,0.3)

        # Reorder the doc
        condensed_doc_with_segments = sort_by_longest_seq(extracted_document)

        summary = ""

        for label,cluster in condensed_doc_with_segments.items():
            paragraph_summary = "".join([sentence.text for sentence in cluster])
            summary+= paragraph_summary
            summary+="\n"

        end = time.time()
        return {"generated_summary": summary, 'SI': cluster_report["SI"],"DBI": cluster_report["DBI"],"DBCV": cluster_report["DBCV"],"time":round((end-start),2)}
    except:
        return {"generated_summary": None, 'SI': None,"DBI": None,"DBCV": None,"time":None}
def dataset_process(row):
    return extraction_pipe(row["report"])

def test_on_dataset(dataset: Dataset, result_file_path:str):
    
    updated_datset = dataset.map(
        dataset_process,
        batched=False,
        )
    results_df = pd.DataFrame.from_dict(updated_datset)
    results_df.to_csv(result_file_path, index=False)
    return results_df

def evaluate_summaries(results_df:pd.DataFrame, file_to_save:str):
    references = results_df["report"]
    predictions = results_df["generated_summary"]

    rouge_metric = load_metric("rouge")
    results = rouge_metric.compute(
    predictions=predictions,
    references=references,
    use_stemmer=True  # Optional: normalize words to their root form
    )

    metrics = []
    for key, value in results.items():
        print(f"{key}: {value.mid}")
        metrics.append([value.mid[0],value.mid[1],value.mid[2]])


    rouge_score_df = pd.DataFrame(metrics,index=["rouge1","rouge2","rougeL","rougeLsum"], columns=["precision","recall","fmeasure"])
    rouge_score_df.to_csv(file_to_save,index=True, header=True)
    return rouge_score_df



    
    



    
    



