import sys
from pacsum import pacsum_extracter
sys.path.insert(0, 'C://GitHub//FYP_ARSynopsis//utils')
from sentence import Sentence
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import time

def text_rank_extractor(sentences:list[Sentence], p_to_select: float) -> list[Sentence]:
    """
    Performs extractive summarization using TextRank algorithm on a given list of sentences.

    Parameters:
        sentences: input sentences/ document for the summarizer
        p_to_select: Propotion of total sentences to be included in the final summary
    Returns:
        List of extracted sentences
    """
    embedding_vectors = [x.embedding for x in sentences]
    similarity_matrix = cosine_similarity(embedding_vectors)
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    ranked_indices= dict(sorted(scores.items(), key=lambda item: item[1],reverse=True))
    extracted_indices = list(ranked_indices.keys())
    extracted_indices = extracted_indices[:int(p_to_select*len(sentences))]
    extracted_indices = sorted(extracted_indices)
    extracted_sentences = [sentences[i] for i in extracted_indices]

    return extracted_sentences

def text_rank_summarizer(clustering:dict[int:list[Sentence]], p_to_select: float) -> dict[int:list[Sentence]]:
    """
    Performs extractive summarization using TextRank algorithm on a given clustering solution.

    Parameters:
        clustering: clustering solution
        p_to_select: propotion of sentences to include in final summary from each group
    Returns:
        Extracted sentences from each cluster along with the cluster labels
    """  
    total_doc_length = 0
    summary_length = 0
    start = time.time()
    extractive_summary = dict()
    for label, cluster in clustering.items():
        extracted_sentences = text_rank_extractor(cluster, p_to_select)
        extractive_summary[label] = extracted_sentences
        total_doc_length += len(cluster)
        summary_length += len(extracted_sentences)
    end = time.time()
    print(f"Extracted {summary_length} sentences from {total_doc_length} belongs to {len(clustering)} clusters in {round((end-start),2)} s.\nAverage summary length: {summary_length/len(clustering)}")
    return extractive_summary

def pacsum_summarizer(clustering:dict[int:list[Sentence]],lambda1:float, lambda2:float, beta:float,  p_to_select: float) -> dict[int:list[Sentence]]:
    """
    Performs extractive summarization using TextRank algorithm on a given clustering solution.

    Parameters:
        clustering: clustering solution
        p_to_select: propotion of sentences to include in final summary from each group
    Returns:
        Extracted sentences from each cluster along with the cluster labels
    """  
    total_doc_length = 0
    summary_length = 0
    start = time.time()
    extractive_summary = dict()
    for label, cluster in clustering.items():
        extracted_sentences = pacsum_extracter(cluster,lambda1,lambda2,beta,p_to_select)
        extractive_summary[label] = extracted_sentences
        total_doc_length += len(cluster)
        summary_length += len(extracted_sentences)
    end = time.time()
    print(f"Extracted {summary_length} sentences from {total_doc_length} belongs to {len(clustering)} clusters in {round((end-start),2)} s.\nAverage summary length: {summary_length/len(clustering)}")
    return extractive_summary



