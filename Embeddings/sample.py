"""
This code consist of the sample use case of converting a pdf into a Sentnece object Array
"""

from preprocessing import get_all_sentences_array
from embeddings import process_sentences_with_positional_encoding_updated

if __name__ == "__main__":
    # Path to the PDF file
    file_path = "sample.pdf" # Change this to the path of your PDF file

    # Load the dataset
    sentences_with_page = get_all_sentences_array(file_path)

    # Extract only the sentence part (ignoring the page number)
    sentences = [sentence_tuple[0] for sentence_tuple in sentences_with_page]

    # Processing the sentence array and converting that to Sentence object array
    sentence_objects = process_sentences_with_positional_encoding_updated(sentences)

    print(f"Processed {len(sentence_objects)} sentences.")
    