import re
import nltk

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