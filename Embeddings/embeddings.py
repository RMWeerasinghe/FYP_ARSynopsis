"""
script to load embeddings
"""
import time
import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import sys
sys.path.insert(0, 'C://GitHub//FYP_ARSynopsis//utils')  # sentence class
from sentence import Sentence

class DocumentLevelPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sentences=5000):
        """
        Document-Level Positional Encoding.
        d_model: dimension of the BERT embeddings (usually 768 for base BERT)
        max_sentences: maximum number of sentences in a document
        """
        super(DocumentLevelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.pe = self._create_positional_encoding(max_sentences, d_model)

    def _create_positional_encoding(self, max_sentences, d_model):
        """
        Helper function to create the positional encoding matrix.
        """
        position = torch.arange(0, max_sentences, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_sentences, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # Add batch dimension

    def forward(self, sentence_position):
        """
        sentence_position: Index of the sentence in the document
        """
        return self.pe[:, sentence_position, :]



def load_bert_model_and_tokenizer():
    """
    Loads pre-trained BERT model and tokenizer.
    Returns both model and tokenizer.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    return tokenizer, bert_model

def tokenize_sentences(sentences, tokenizer):
    """
    Tokenizes sentences and prepares inputs for BERT.
    Returns a list of tokenized inputs.
    """
    return [tokenizer(sentence, return_tensors='pt', padding=True, truncation=True) for sentence in sentences]

def add_positional_encoding_to_embeddings(bert_model, doc_pos_encoder, inputs):
    """
    Processes sentences by getting BERT embeddings, adding document-level positional encoding.
    Returns sentence embeddings with and without positional encoding.
    """
    sentence_embeddings_without_pos_encoding = []
    sentence_embeddings_with_pos_encoding = []

    for idx, input_dict in enumerate(inputs):
        # Pass the sentence input through BERT to get token embeddings, including attention_mask
        token_embeddings = bert_model(input_dict['input_ids'], attention_mask=input_dict['attention_mask']).last_hidden_state

        # Extract sentence embedding without positional encoding (using [CLS] token)
        sentence_embeddings_without_pos_encoding.append(token_embeddings[:, 0, :])

        # Add document-level positional encoding
        doc_pos_encoding = doc_pos_encoder(idx).unsqueeze(1)  # Add sequence dimension
        modified_embeddings = token_embeddings + doc_pos_encoding

        # Aggregate sentence embedding (e.g., taking the [CLS] token's embedding)
        cls_embedding = modified_embeddings[:, 0, :]  # Extract the [CLS] token representation
        sentence_embeddings_with_pos_encoding.append(cls_embedding)

    return sentence_embeddings_without_pos_encoding, sentence_embeddings_with_pos_encoding

# This commented function is used when using BERT pretrained model
# def process_sentences_with_positional_encoding(sentences):
#     """
#     Main function to process an array of sentences by:
#     - Loading BERT model and tokenizer
#     - Tokenizing sentences
#     - Applying document-level positional encoding
#     - Returning sentence embeddings with and without positional encoding

#     Parameters:
#     sentences: List of sentences to process.

#     Returns:
#     Tuple containing:
#       - sentence_embeddings_without_pos_encoding
#       - sentence_embeddings_with_pos_encoding
#     """
#     # Load BERT model and tokenizer
#     tokenizer, bert_model = load_bert_model_and_tokenizer()

#     # Initialize document-level positional encoder
#     doc_pos_encoder = DocumentLevelPositionalEncoding(d_model=768)

#     # Tokenize sentences
#     inputs = tokenize_sentences(sentences, tokenizer)

#     # Process sentences, adding document-level positional encoding
#     sentence_embeddings_without_pos_encoding, sentence_embeddings_with_pos_encoding = add_positional_encoding_to_embeddings(
#         bert_model, doc_pos_encoder, inputs
#     )

#     # Create Sentence objects for each sentence
#     sentence_objects = []
#     for idx, sentence in enumerate(sentences):
#         # Create Sentence object with sentence ID, text, and positional encoded embeddings
#         sentence_obj = Sentence(id=idx, sentence=sentence, embeddings=sentence_embeddings_with_pos_encoding[idx])
#         sentence_objects.append(sentence_obj)

#     return sentence_objects



def process_sentences_with_positional_encoding_updated(sentences):
    """
    Main function to process an array of sentences by:
    - Loading SentenceTransformer model
    - Tokenizing sentences
    - Applying document-level positional encoding
    - Returning sentence embeddings with positional encoding
    """

    start = time.time()
    # Load SentenceTransformer model
    model = SentenceTransformer('sentence-transformers/xlm-r-bert-base-nli-mean-tokens')

    # Extract sentences from tuples
    sentence_texts = [sentence for sentence in sentences]

    # Generate sentence embeddings
    sentence_embeddings = model.encode(sentence_texts, convert_to_tensor=True)

    # Initialize document-level positional encoder
    doc_pos_encoder = DocumentLevelPositionalEncoding(d_model=sentence_embeddings.shape[1])

    # Apply positional encoding
    sentence_embeddings_with_pos = []

    # Apply positional encoding and create Sentence objects
    sentence_objects = []

    for idx, sentence_embedding in enumerate(sentence_embeddings):
        pos_encoding = doc_pos_encoder(idx).squeeze(0)  # Get positional encoding for current sentence
        modified_embedding = sentence_embedding + pos_encoding

        sentence_embeddings_with_pos.append(modified_embedding)

        # Create a unique UUID for each sentence
        sentence_id = idx  # Generate a unique UUID and convert it to a string

        # Create Sentence object with sentence ID, text, and positional encoded embeddings
        sentence_obj = Sentence(index = sentence_id, text=sentence_texts[idx])
        sentence_obj.set_embedding(modified_embedding)
        sentence_objects.append(sentence_obj)

    end = time.time()
    print(f"Obtained Positional Encoded Embeddings in {round((end-start),2)}")
    return sentence_objects


# Add function to get embeddings without positional embedding

def process_sentences_without_positional_encoding_updated(sentences):
    """
    Main function to process an array of sentences by:
    - Loading SentenceTransformer model
    - Tokenizing sentences
    - Returning sentence embeddings with positional encoding
    """
    # Load SentenceTransformer model
    model = SentenceTransformer('sentence-transformers/xlm-r-bert-base-nli-mean-tokens')

    # Extract sentences from tuples
    sentence_texts = [sentence for sentence in sentences]

    # Generate sentence embeddings
    sentence_embeddings = model.encode(sentence_texts, convert_to_tensor=True)

    sentence_objects = []

    for idx, sentence_embedding in enumerate(sentence_embeddings):

        # Create a unique UUID for each sentence
        sentence_id = idx  # Generate a unique UUID and convert it to a string

        # Create Sentence object with sentence ID, text, and embeddings
        sentence_obj = Sentence(index = sentence_id, text=sentence_texts[idx])
        sentence_obj.set_embedding(sentence_embedding)
        sentence_objects.append(sentence_obj)

    return sentence_objects


# Embedding creation function for comparative analysis - create two types in one attepmt

def process_sentences_and_encode(sentences):
    """
    Main function to process an array of sentences by:
    - Loading SentenceTransformer model
    - Tokenizing sentences
    - Applying document-level positional encoding
    - Returning sentence embeddings with positional encoding
    """
    # Load SentenceTransformer model
    model = SentenceTransformer('sentence-transformers/xlm-r-bert-base-nli-mean-tokens')

    # Extract sentences from tuples
    sentence_texts = [sentence for sentence in sentences]

    # Generate sentence embeddings
    sentence_embeddings = model.encode(sentence_texts, convert_to_tensor=True)

    # Initialize document-level positional encoder
    doc_pos_encoder = DocumentLevelPositionalEncoding(d_model=sentence_embeddings.shape[1])

    # Apply positional encoding
    sentence_embeddings_with_pos = []

    # Apply positional encoding and create Sentence objects
    sentence_objects_pos = []
    sentence_objects_without_pos = []

    for idx, sentence_embedding in enumerate(sentence_embeddings):
        pos_encoding = doc_pos_encoder(idx).squeeze(0)  # Get positional encoding for current sentence
        modified_embedding = sentence_embedding + pos_encoding

        sentence_embeddings_with_pos.append(modified_embedding)

        # Create a unique UUID for each sentence
        sentence_id = idx  # Generate a unique UUID and convert it to a string

        # Create Sentence object with sentence ID, text, and positional encoded embeddings
        sentence_obj_pos = Sentence(index = sentence_id, text=sentence_texts[idx])
        sentence_obj_pos.set_embedding(modified_embedding)
        sentence_objects_pos.append(sentence_obj_pos)

        sentence_obj_without_pos = Sentence(index = sentence_id, text=sentence_texts[idx])
        sentence_obj_without_pos.set_embedding(sentence_embedding)
        sentence_objects_without_pos.append(sentence_obj_without_pos)
        

    return sentence_objects_without_pos, sentence_objects_pos
    

