"""
script to load embeddings
"""

import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
from sentence_transformers import SentenceTransformer

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