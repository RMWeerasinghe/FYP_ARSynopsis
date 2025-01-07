from numpy import ndarray
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sentence import Sentence
from IPython.display import display, HTML
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import sys
sys.path.insert(0, 'C://GitHub//FYP_ARSynopsis//utils')
from section_summary import SectionSummary

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

from nltk.tokenize import sent_tokenize

def display_document_with_clusters(document: list[Sentence]):
    '''
    Display the complete PDF report annotating sentences belongs to different clusters in different colours
    '''
    colors = ['#FF5733', '#33C3FF', '#FF33F6', '#75FF33', '#FF3380', '#33FFBD', '#FF9933',
               '#3366FF', '#FF3333', '#33FF57', '#FF33A6', '#33A6FF', '#FF66FF', '#66FF33', 
               '#FFCC33', '#6633FF', '#FF6633', '#33FF66', '#FF33CC', '#33FF99', '#FF3366',
                 '#33FFCC', '#FF66CC', '#6699FF', '#FF9933', '#33FF80', '#FF00FF', '#00CCFF', 
                 '#FF6600', '#33CC33']
    html_text = ""
    for sentence in document:
        html_str = f'<span style="color: {colors[sentence.cluster]};">{sentence.text}</span>'
        html_text = " ".join([html_text,html_str])
    
    display(HTML(html_text))

def get_longest_consec_seq(cluster : list[Sentence]) -> int:
    '''
    Helper function to get the starting index of the longest cosecutive sub sequence of the extracted cluster.

    Parameters:
        cluster: list of Sentence objects
    Returns: Starting index of the longest consecutive subsequence
    '''
    longest_start = None
    max_length = 0
    num_set = set([sent.index for sent in cluster])

    for sent in cluster:
        if sent.index - 1 not in num_set:
            current_start = sent.index
            length = 1

            while current_start + length in num_set:
                length +=1
            
            if length > max_length:
                max_length = length
                longest_start = sent.index

    return longest_start


def sort_by_longest_seq(clustering:dict[int:list[Sentence]]) -> dict[int:list[Sentence]]:
    '''
    Re-order the clusters according to the start index of the longest cosecutive sub sequence to provide 
    a meaningful organization of the condensed document.

    Parameters:
        clustering: A clustering solution, preferably the extracted document
    Returns:
        Re-ordered version of the document
    '''
    start_indices = {label: get_longest_consec_seq(cluster) for label,cluster in clustering.items()}
    sorted_clustering = dict(sorted(clustering.items(), key=lambda item: start_indices[item[0]]))

    sorted_clustering_text = list(sorted_clustering.values())


    return sorted_clustering_text

def get_condensed_report(clustering_after_extraction:dict[int:list[Sentence]]) -> list[str]:
    """
    Prepares the condensed document after extraction process is done. Outputs list of paragraphs.

    parameters: 
            clustering_after_extraction - Extractive summaries for each cluster
    Returns:
            List of paragraphs
    """
    re_ordered_doc = sort_by_longest_seq(clustering_after_extraction)

    condensed_report = [" ".join([sent.text for sent in section]) for section in re_ordered_doc]

    return condensed_report

def create_output(section_list:list[str],summary_list:list[str]):
    model = SentenceTransformer('sentence-transformers/xlm-r-bert-base-nli-mean-tokens')
    """
    Prepares the annual report summary to get the final output. Returns the list of SectionSummary obejcts,
    where each SectionSummary is consisted of
            1. Section ID
            2. Final Summary of the System
            3. List of mapping: list[i] = x -> i th sentence of the summary is related to the sentence x
    
    Parameters:
        section_list: list of strings generated 
    """
    start = time.time()
    summarized_report = []
    for index in range(len(section_list)):
        content_sentences = sent_tokenize(section_list[index])
        content_embeddings =model.encode(content_sentences, convert_to_numpy=True) 
        summary_embeddings = model.encode(sent_tokenize(summary_list[index]), convert_to_numpy=True)

        similarity_matrix = cosine_similarity(summary_embeddings,content_embeddings)
        similar_indices = similarity_matrix.argmax(axis=1)

        mapping_list = [content_sentences[k] for k in similar_indices]

        section_summary = SectionSummary(index)
        section_summary.add_summary(summary_list[index])
        section_summary.set_mapping(mapping_list)

        summarized_report.append(section_summary)
    end = time.time()

    print(f"Prepared summarized report from generated summaries in {round((end-start),2)}s")
    return summarized_report
