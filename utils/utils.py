from numpy import ndarray
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sentence import Sentence
from IPython.display import display, HTML


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

    return sorted_clustering
