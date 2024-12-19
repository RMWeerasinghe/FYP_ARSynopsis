# imports
import time
import os
import sys
import pandas as pd
import nltk
sys.path.insert(0, 'C://GitHub//FYP_ARSynopsis//Embeddings')
from embeddings import process_sentences_with_positional_encoding_updated, process_sentences_and_encode
sys.path.insert(0, 'C://GitHub//FYP_ARSynopsis//Clustering')
from clustering import get_feature_matrix, evaluate, reduce_and_cluster
from sklearn.metrics.cluster import v_measure_score

nltk.download('punkt')
nltk.download('punkt_tab')

def read_document_and_prepare_dataset(file_path:str, seperater: str)->tuple[list[str],list[int],int]:
    """
    Read each text file in the given path and then return the lsit of sentences in the document and their corresponding cluster label (segement index)

    Arguments:
        file_path: path of the text file
        seperater : segment boundary style
    Returns:
        list of sentences 
        Ground truth labels
        number of sentences
    """
    start = time.time()
    sentence_list = []
    y_true = []
    with open(file_path,"r") as f:

                raw_text = f.read()
                raw_sentence_list = nltk.tokenize.sent_tokenize(raw_text)
                i = -1
                for s in raw_sentence_list:
                    if s == "***LIST***":
                        # ignore
                        continue
                    if seperater in s:
                        i += 1
                        # for wiki-50 and wiki-727
                        parts = s.split(",", maxsplit=1)
                        if len(parts) > 1 and parts[1].strip():  # Ensure text exists after the pattern
                            sentence_list.append(parts[-1].strip())  # Add the text to the list
                            y_true.append(i)
                    else:
                        sentence_list.append(s)
                        y_true.append(i)
    end = time.time()
    if len(sentence_list) == len(y_true):
        print(f"Extracted {len(sentence_list)} sentences with {len(y_true)} labels in {round((end-start),2)}s.")
        return sentence_list, y_true,len(sentence_list)
    else:
        raise IndexError


def get_embedding_matrices(sentence_list:list[str],control_exp:bool = True):
     
    """
     Create embedding matrices for a given sentence list.
     If control_exp = True, returns two independent embedding matrices with and without positional encoding.
     Else returns positional encoded embeddings.

    """

    if control_exp:
        document_without_pos, document_with_pos = process_sentences_and_encode(sentence_list)
        embedding_without_pos = get_feature_matrix(document_without_pos)
        embedding_with_pos = get_feature_matrix(document_with_pos)

        return embedding_without_pos, embedding_with_pos

    document_with_pos = process_sentences_with_positional_encoding_updated(sentence_list)
    embedding_with_pos = get_feature_matrix(document_with_pos)
    return embedding_with_pos

def evaluate_clustering(X, y_true,y_pred):
    """
    Evaluate the clustering solution using V Measure and Internal Evaluation metrics
    """
    eval_report = evaluate(X,y_pred)
    v_measure = v_measure_score(y_true,y_pred)

    eval_report["VM"] = v_measure

    return eval_report

def test(folder_path:str, test_set_name:str,seperator:str = "==="):
    start = time.time()
    test_files = os.listdir(folder_path)
    count = 0
    df = pd.DataFrame(
         columns=["documen_name","num_sentences","num_segments_true","num_clusters","VM","SI","DBI","DBCV","time","VM_pos","SI_pos","DBI_pos","DBCV_pos","time_pos"])
    for file in test_files:
        try:
            
            file_path = os.path.join(folder_path,file)
            start1 = time.time()
            sentence_list, y_true, num_sentences = read_document_and_prepare_dataset(file_path,seperator)
            embedding_without_pos, embedding_with_pos = get_embedding_matrices(sentence_list)

            #without_pos
            cluster_labels_without_pos, cluster_record_without_pos = reduce_and_cluster(embedding_without_pos,10)
            eval_without_pos = evaluate_clustering(embedding_without_pos,y_true,cluster_labels_without_pos)

            # with pos
            cluster_labels_with_pos, cluster_record_with_pos = reduce_and_cluster(embedding_with_pos,10)
            eval_with_pos = evaluate_clustering(embedding_with_pos,y_true,cluster_labels_with_pos)
            
            df.loc[len(df)] = [file,num_sentences,len(set(y_true)),cluster_record_with_pos["n_clusters"],
                            eval_without_pos["VM"],eval_without_pos["SI"],eval_without_pos["DBI"],eval_without_pos["DBCV"], cluster_record_without_pos["time"],
                            eval_with_pos["VM"],eval_with_pos["SI"],eval_with_pos["DBI"],eval_with_pos["DBCV"], cluster_record_with_pos["time"]]
            end1 = time.time()

            print(f"Processed {file} in {round((end1-start1),2)}s.")

            count += 1
        except Exception as e:
            print(f"Error occured while processing {file} {e}")        
            continue

    df.to_csv(f"test_results/{test_set_name}.csv",header=True, index=False)   
    end = time.time()
    print(f"Evaluted {count} text documents in {round((end-start),2)}s.")  

    return df


    


    