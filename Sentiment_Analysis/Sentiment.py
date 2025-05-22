import pandas as pd
import sys
sys.path.insert(0, 'C://GitHub//FYP_ARSynopsis//utils')
from sentence import Sentence
from transformers import pipeline
import torch


MODEL_NAME = "ProsusAI/finbert"
MODEL = pipeline(task='sentiment-analysis', model=MODEL_NAME, max_length=512, truncation=True, device="cuda" if torch.cuda.is_available() else "cpu")


def create_data_for_sentiment_analysis(extracted_document:dict[int:list[Sentence]])->pd.DataFrame:
    """
    Create pandas dataframe object to perform sentiment analysis
    """
    return pd.DataFrame(
    [(sentence.text, section_id) for section_id, sentences in extracted_document.items() for sentence in sentences],
    columns=["sentence", "section_id"]
    )

def get_sentiments(input_df:pd.DataFrame):
    """
    Perform sentiment analysis
    """
    output = MODEL(input_df["sentence"].to_list())
    output_df = pd.DataFrame(output)
    input_df["label"] = output_df["label"]
    input_df["score"] = output_df["score"]

    return input_df

def create_sentiment_output(input_df:pd.DataFrame):

    sections = input_df["section_id"].unique() 
    sentiment_results = dict()
    sentiment_propotions = dict()

    for sec in sections:
        section_df = input_df[input_df["section_id"]==sec]
        total = section_df.shape[0]
        pos_df = section_df.loc[section_df["label"] == "positive", ["sentence","score"]].sort_values(by = "score", ascending = False)
        neg_df = section_df.loc[section_df["label"] == "negative", ["sentence","score"]].sort_values(by = "score", ascending = False)
        n_pos = pos_df.shape[0]
        n_neg = neg_df.shape[0]


        sentiment_results[sec] = {"pos":pos_df["sentence"].to_list()[0:4],"neg":neg_df["sentence"].to_list()[0:4]}
        sentiment_propotions[sec] = {"pos":round(n_pos/total,2),"neg":round(n_neg/total,2),"neu":round(1 - (n_pos+n_neg)/total,2)}
    

    return sentiment_results, sentiment_propotions


def sentiment_analyser(extracted_document:dict[int:list[Sentence]]):

    input_df = create_data_for_sentiment_analysis(extracted_document)
    input_df = get_sentiments(input_df)
    sentiment_results, sentiment_propotions = create_sentiment_output(input_df)
    return sentiment_results, sentiment_propotions
