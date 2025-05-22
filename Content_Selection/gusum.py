"""
@InProceedings{gokhan-smith-lee:2022:textgraphs,
  author    = {Gokhan, Tuba  and  Smith, Phillip  and  Lee, Mark},
  title     = {GUSUM: Graph-based Unsupervised Summarization Using Sentence Features Scoring and Sentence-BERT},
  booktitle      = {Proceedings of TextGraphs-16: Graph-based Methods for Natural Language Processing},
  month          = {October},
  year           = {2022},
  address        = {Gyeongju, Republic of Korea},
  publisher      = {Association for Computational Linguistics},
  pages     = {44--53},
  abstract  = {Unsupervised extractive document summarization aims to extract salient sentences from a document without requiring a labelled corpus. In existing graph-based methods, vertex and edge weights are usually created by calculating sentence similarities. In this paper, we develop a Graph-Based Unsupervised Summarization(GUSUM) method for extractive text summarization based on the principle of including the most important sentences while excluding sentences with similar meanings in the summary. We modify traditional graph ranking algorithms with recent sentence embedding models and sentence features and modify how sentence centrality is computed. We first define the sentence feature scores represented at the vertices, indicating the importance of each sentence in the document. After this stage, we use Sentence-BERT for obtaining sentence embeddings to better capture the sentence meaning. In this way, we define the edges of a graph where semantic similarities are represented. Next we create an undirected graph that includes sentence significance and similarities between sentences. In the last stage, we determine the most important sentences in the document with the ranking method we suggested on the graph created. Experiments on CNN/Daily Mail, New York Times, arXiv, and PubMed datasets show our approach achieves high performance on unsupervised graph-based summarization when evaluated both automatically and by humans.},
  url       = {https://aclanthology.org/2022.textgraphs-1.5}
}

Implemeneted refering GUSUM implementation at https://github.com/tubagokhan/GUSUM/tree/main
"""

import sys
import numpy as np
from sentence_transformers import SentenceTransformer
sys.path.insert(0, 'C://GitHub//FYP_ARSynopsis//utils')
from sentence import Sentence
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('averaged_perceptron_tagger_eng')

from nltk import sent_tokenize, word_tokenize, PorterStemmer

# Sentence features

## Sentence Length



def length_score(paragraph:list[Sentence]) -> np.array:
    
    word_counts = np.array([x.word_count() for x in paragraph])
    length_scores = word_counts/np.max(word_counts)
    return length_scores

## Position

def position_score(paragraph:list[Sentence]):
    n = len(paragraph)
    position_scores = np.array(range(1,n+1))
    position_scores = (n - position_scores)/n
    position_scores[0] = position_scores[-1] = 1
    return position_scores

## Proper Noun Count and Numerical Count

def count_tags(sent:str):
    
    text  = word_tokenize(sent)
    if len(text)>0:
      tags = nltk.pos_tag(text)
      proper_noun_count = 0
      numerical_count = 0

      for word, tag in tags:
          if tag == "NNP" :# Proper noun
              proper_noun_count+=1
          elif tag == "CD": # Numerical
              numerical_count+=1
      return proper_noun_count/len(text), numerical_count/len(text)
    
    return 0,0

def tag_scores(paragraph:list[Sentence]):
    proper_noun_scores = []
    numerical_scores = []

    for sentence in paragraph:
        pnouns, numerals = count_tags(sentence.text)
        proper_noun_scores.append(pnouns)
        numerical_scores.append(numerals)
      
    return np.array(proper_noun_scores), np.array(numerical_scores)


def feature_score(paragraph:list[Sentence]):
    position_scores = position_score(paragraph)
    length_scores = length_score(paragraph)
    proper_noun_scores, numerical_scores = tag_scores(paragraph)

    feature_scores = position_scores + length_scores + proper_noun_scores + numerical_scores

    return feature_scores


def calculate_cosine_similarity_matrix(paragraph:list[Sentence]):
    embedding_vectors = np.array([x.pos_embedding.cpu().detach().numpy() for x in paragraph])
    similarity_matrix = cosine_similarity(embedding_vectors)
    total_similarity_score_per_sent = np.sum(similarity_matrix, axis=0)
    return total_similarity_score_per_sent

def rank_sentences(paragraph:list[Sentence]):
    feature_scores = feature_score(paragraph)
    total_similarity_score_per_sent = calculate_cosine_similarity_matrix(paragraph)
    rank_scores = feature_scores * total_similarity_score_per_sent
    sentences = np.array(paragraph)
    ranked_sentences = sentences[np.argsort(rank_scores)[::-1]].tolist()
    return ranked_sentences

def gusum_extracter(paragraph:list[Sentence], p:float):
    ranked_sentences = rank_sentences(paragraph)
    k = min(int(len(paragraph)*p),80)
    top_k_sentences = ranked_sentences[:k]
    sorted_top_k = sorted(top_k_sentences, key=lambda x: x.index)
    return sorted_top_k







    
    
    