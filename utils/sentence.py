class Sentence:
    """
    A new data structure to represent a sentence in a PDF.
    E.g: 
     Our stock price may be highly volatile and could decline in value.
     Future sales of our common stock will result in dilution to our common stockholders.

    Attributes:
    ---------------------------
    _index : int
        A unique identifier for the sentence object refers to the position of the sentence in PDF
    _text : string
        The actual sentence in human language
    _embedding: array
        Embedding vector of the sentence
    _cluster: int
        Cluster label of the sentence. Initialize to -1 indicating complete PDF
    
    Methods:
    --------------------------
    set_embedding(embedding):
        Add generated embedding of the sentence
    set_cluster_label(int):
        Add cluster label after clustering
    word_count():
        Gives the word count of the sentence
    
    """

    def __init__(self,index,text) -> None:
        self._index = index
        self._text = text
        self._embedding = None
        self._cluster = -1
    
    def set_embedding(self,embedding) -> None:
        self._embedding = embedding
    
    def set_cluster_label(self,label) -> None:
        self._cluster = label
    def word_count(self) -> int:
        return (len(self._text.split(" ")))