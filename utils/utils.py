from numpy import ndarray
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pandas import DataFrame
from sentence import Sentence
from IPython.display import display, HTML


def display_document_with_clusters(document: list[Sentence]):
    '''
    Display the complete PDF report annotating sentences belongs to different clusters in different colours
    '''
    colors = ['#FF5733', '#33C3FF', '#FF33F6', '#75FF33', '#FF3380', '#33FFBD', '#FF9933',
               '#3366FF', '#FF3333', '#33FF57', '#FF33A6', '#33A6FF', '#FF66FF', '#66FF33', 
               '#FFCC33', '#6633FF', '#FF6633', '#33FF66', '#FF33CC', '#33FF99', '#FF3366',
                 '#33FFCC', '#FF66CC', '#6699FF', '#FF9933', '#33FF80', '#FF00FF', '#00CCFF', 
                 '#FF6600', '#33CC33']
    html_text = ""
    for sentence in document:
        html_str = f'<span style="color: {colors[sentence.cluster]};">{sentence.text}</span>'
        html_text = " ".join([html_text,html_str])
    
    display(HTML(html_text))


    
