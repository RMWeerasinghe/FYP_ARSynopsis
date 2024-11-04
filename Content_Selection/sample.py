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
from extractive_summarizer import text_rank_summarizer

if __name__ == "__main__":

    sample_pdf = r"Sample_Reports\tesla_report.pdf"

    # Extract document content
    sentence_array = get_all_sentences_array(sample_pdf)
    sentence_array = [sentence_tuple[0] for sentence_tuple in sentence_array]

    # Get embeddings with positional encoding
    document = process_sentences_with_positional_encoding_updated(sentence_array)

    # K-Means Clustering with dimensionality reduction and parameter tuning
    clustering, cluster_report = k_means_cluster_document(document, n_trials= 3)

    # extractive summarization using Text Rank
    extracted_document = text_rank_summarizer(clustering, 0.3)

    # Reorder the doc
    condensed_doc_with_segments = sort_by_longest_seq(extracted_document)

    print(f"Re-ordering of cluster: {condensed_doc_with_segments.keys()}")

    # Sample output format

    # {cluster_label 1 (int) : 
    #     [Sentence Object 1, Sentence Object 2, Sentence Object 3, ....],
    # cluster_label 2 (int) : 
    #     [Sentence Object 4, Sentence Object 5, Sentence Object 6, ....],
    # ...
    #     }

    # Sentence Object : See Utils/sentence.py
    # Sentece:
    #     - index
    #     - embedding
    #     - text
    #     - cluster
