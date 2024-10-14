"""
script to load embeddings
"""

import torch
from transformers import BertTokenizer, BertModel

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