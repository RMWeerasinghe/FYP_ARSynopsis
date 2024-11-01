import re
import nltk
from pypdf import PdfReader

nltk.download('punkt')

# Function to remove email addresses from text
def remove_emails(text):
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    return re.sub(email_pattern, '', text) 


# Function to remove telephone numbers from text
def remove_phone_numbers(text):
    phone_pattern = r'\(?\b[0-9]{3}[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'
    return re.sub(phone_pattern, '', text)


# Function to split text into sentences while avoiding certain splits
def split_into_sentences(text):
    # Replace newline characters with spaces
    clean_text = text.replace("\n", " ")

    # Further clean up, e.g., stripping extra spaces
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    # Use NLTK's sentence tokenizer for more robust splitting
    sentences = nltk.tokenize.sent_tokenize(clean_text)

    return sentences

# Function to remove URLs from text
def remove_urls(text):
    # Regular expression pattern for matching URLs
    url_pattern = r'https?://(?:www\.)?\S+|www\.\S+'
    
    # Explanation:
    # - `https?`: Matches "http" or "https". The `s?` makes the "s" optional.
    # - `://`: Matches the literal "://" in URLs.
    # - `(?:www\.)?`: Non-capturing group to match "www." optionally.
    #     - `www\.`: Matches "www." literally.
    #     - `?`: Makes the preceding "www." part optional, allowing for URLs with or without "www".
    # - `\S+`: Matches one or more non-whitespace characters, which covers the rest of the URL.
    # - `|`: OR operator, allowing for an alternative pattern on the right side.
    # - `www\.\S+`: Matches URLs starting with "www." but without "http" or "https".
    # - This combination will match URLs starting with "http://", "https://", or just "www.".
    
    # Substitute all URLs in the text with an empty string to remove them
    return re.sub(url_pattern, '', text)


def get_all_sentences_array(file_path):


    # Initialize the PdfReader object to read the PDF file from the given file path
    reader = PdfReader(file_path)

    # Create an empty list to store all extracted sentences
    all_sentences = []

    # Get the total number of pages in the PDF document
    number_of_pages = len(reader.pages)
    print(f"Total Number of Pages: {number_of_pages}")

    # Loop through all the pages in the PDF (from the first page to the last)
    for page_num in range(number_of_pages):
        # Get the current page
        page = reader.pages[page_num]

        # Extract the text content from the current page
        text = page.extract_text()
        # print(f"Processing Page {page_num + 1}...")

        # Remove any emails and phone numbers from the text for privacy or cleanup
        text = remove_emails(text)
        text = remove_phone_numbers(text)
        text = remove_urls(text)

        # Split the cleaned text into individual sentences
        sentences = split_into_sentences(text)

        # Append each sentence along with its corresponding page number to the list
        all_sentences.extend([(sentence, page_num + 1) for sentence in sentences])

    # Return the final list of all sentences along with their page numbers
    return all_sentences
