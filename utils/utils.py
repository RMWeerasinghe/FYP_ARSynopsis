from numpy import ndarray
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pandas import DataFrame
from sentence import Sentence

def visualize_document(document:list[Sentence],with_label = False) -> plt:
    """Function to plot document in a 2D plot using TSNE"""
    
