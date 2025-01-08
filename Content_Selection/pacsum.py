import sys
import numpy as np
from sentence_transformers import SentenceTransformer
sys.path.insert(0, 'C://GitHub//FYP_ARSynopsis//utils')
from sentence import Sentence

model = SentenceTransformer('sentence-transformers/xlm-r-bert-base-nli-mean-tokens')

def get_document_vectors(document:list[Sentence]) -> list[np.array]:
    """
    Get document vectors for the given list of sentences

    Parameters:
        document: list of sentences
    Returns:
        List of document vectors for the given list of sentences
        document_vectores[i] = sentence[i].embedding
    """
    document_vectors = np.array([sentence.embedding for sentence in document])
    return document_vectors

def calculate_similarity_matrix(document:list[Sentence]) -> list[list[float]]:
    """
    Calculate similarity matrix for the given list of sentences

    Parameters:
        document: list of sentences
    Returns:
        Similarity matrix for the given list of sentences measure using pairwise dot product.
        similarity between v_i and v_j, e_ij = v_i.T * v_j
    """

    document_vectors = get_document_vectors(document)
    similarity_matrix = np.dot(document_vectors, document_vectors.T)
    return similarity_matrix

    

def pacsum_extracter(document:list[Sentence], lambda1:float, lambda2:float, beta:float, p_to_select:float) -> list[Sentence]:
    """
    Extractive summarization using PACSum algorithm

    Parameters:
        document: list of sentences to be summarized
        lambda1, lambda2: parameter for the sentence similarity formulation
        beta: similarity threshold parameter
        p_to_select: propotion of sentences to be extracted.
    Returns:
        List of extracted sentences
    """
    similarity_matrix = calculate_similarity_matrix(document)
    # normalize similarity matrix
    norm_matrix = similarity_matrix + (beta - 1)*np.min(similarity_matrix) - beta*np.max(similarity_matrix)
    norm_matrix [norm_matrix < 0] = 0
    centrality_array = [0]*len(document)
    for i in range(len(document)):
        lambda1_sum = norm_matrix[i,:i].sum()
        lambda2_sum = norm_matrix[i,i+1:].sum()
        centrality = lambda1*lambda1_sum + lambda2*lambda2_sum
        centrality_array[i] = centrality
    # sort the sentences based on centrality
    n = int(len(document)*p_to_select)
    sorted_indices = np.argsort(centrality_array)[::-1][:n]
    sorted_indices.sort()
    extracted_document = []
    for k in sorted_indices:
        extracted_document.append(document[k])
    return extracted_document
    
    