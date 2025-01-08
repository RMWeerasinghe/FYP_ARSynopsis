class Sentence:
    """
    A new data structure to represent a sentence in a PDF.
    E.g: 
     Our stock price may be highly volatile and could decline in value.
     Future sales of our common stock will result in dilution to our common stockholders.

    Attributes:
    ---------------------------
    index : int
        A unique identifier for the sentence object refers to the position of the sentence in PDF
    text : string
        The actual sentence in human language
    embedding : array
        Sentence BERT embedding of the sentence
    pos_embedding: array
        Positional Encoded Embedding vector of the sentence
    cluster: int
        Cluster label of the sentence. Initialize to -1 indicating complete PDF
    
    Methods:
    --------------------------
    set_embedding(embedding):
        Add generated SentenceBERT embedding of the sentence
    set_pos_embedding(embedding):
        Add generated positional encoded embedding of the sentence
    set_cluster_label(int):
        Add cluster label after clustering
    word_count():
        Gives the word count of the sentence
    
    """

    def __init__(self,index,text) -> None:
        self.index = index
        self.text = text
        self.embedding = None
        self.pos_embedding = None
        self.cluster = -1
    
    def set_embedding(self,embedding) -> None:
        self.embedding = embedding

    def set_pos_embedding(self,pos_embedding) -> None:
        self.pos_embedding = pos_embedding
    
    def set_cluster_label(self,label) -> None:
        self.cluster = label

    def word_count(self) -> int:
        return (len(self.text.split(" ")))
    
    def get_id(self) -> int:
        return self.index