import sys

sys.path.insert(0, 'C://GitHub//FYP_ARSynopsis//utils')
from sentence import Sentence
from utils import display_document_with_clusters, get_condensed_report

sys.path.insert(0, 'C://GitHub//FYP_ARSynopsis//Embeddings')
from preprocessing import get_all_sentences_array
from embeddings import process_sentences_with_positional_encoding_updated

sys.path.insert(0, 'C://GitHub//FYP_ARSynopsis//Clustering')
from clustering import k_means_cluster_document

sys.path.insert(0, 'C://GitHub//FYP_ARSynopsis//Content_Selection')
from extractive_summarizer import text_rank_summarizer, pacsum_summarizer

if __name__ == "__main__":

    sample_pdf = r"Sample_Reports\tesla_report.pdf"

    # Extract document content
    sentence_array = get_all_sentences_array(sample_pdf)
    sentence_array = [sentence_tuple[0] for sentence_tuple in sentence_array]

    # Get embeddings with positional encoding
    document = process_sentences_with_positional_encoding_updated(sentence_array)

    # K-Means Clustering with dimensionality reduction and parameter tuning
    clustering, cluster_report = k_means_cluster_document(document,120)

    # extractive summarization using Text Rank
    #extracted_document = text_rank_summarizer(clustering, 0.3)

    # extractive summarization using PacSum
    extracted_document = pacsum_summarizer(clustering,-2,1,0.6,0.2)

    # Reorder the doc
    condensed_report = get_condensed_report(extracted_document)


    for section in condensed_report:
        print(f"=========================================================")
        print(section)

    # Sample output format

    #[ 
    #     [paragraph 1, paragraph 2, paragraph 3, ....],
    # cluster_label 2 (int) : 
    #     [paragraph 4, paragraph 5, paragraph 6, ....],
    # ...
    #     ]

    # paragraph : All text content belongs to a section/ cluster
 