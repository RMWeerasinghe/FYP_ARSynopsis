"""
Script to cluster the data
"""
# imports
import sys
import numpy as np
import umap
import optuna
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import time
from permetrics import ClusteringMetric

# Data Structures
sys.path.insert(0, 'C://GitHub//FYP_ARSynopsis//utils')
from sentence import Sentence

def get_feature_matrix(document:list[Sentence])-> np.array:
    '''
        Returns M x N feature matrix (Embedding Matrix) for a given list of sentence objects,
        where
         M: Number of Sentences in the document
         N: Embedding Dimention
    '''
    feature_mat = np.array([x.embedding for x in document])
    return feature_mat

def reduce_and_cluster(feature_mat:np.array, n_trials:int = 10, n_clusters_min:int = 10, n_clusters_max: int = 20):
    '''
    Apply UMAP dimensionality reduction and then clustered using k-means clustering. Hyperparameters of UMAP and K-Means
    are selected using Optuna Tuning job

    Embedding Matrix -> Standardization -> UMAP -> Clustering

    '''
    start = time.time()
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(feature_mat)

    def objective(trial):
        # Suggest hyperparameters

        # Remember to change
        n_neighbors = trial.suggest_categorical('n_neighbors', list(range(5,50,3)))
        min_dist = trial.suggest_float('min_dist', 0.0, 0.99)
        n_components = trial.suggest_categorical('n_components', [48,96,128,256,384])
        num_clusters = trial.suggest_int('n_clusters',n_clusters_min,n_clusters_max)

        # Initialize UMAP
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric= 'cosine',
            random_state=42,
            n_jobs= 1,
        )
        
        # Reduce dimensionality
        embedding = reducer.fit_transform(embeddings_scaled)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init = "auto")
        cluster_labels = kmeans.fit_predict(embedding)

        cluster_count = len(np.unique(cluster_labels))
        
        # Calculate Silhouette Score
        if len(np.unique(cluster_labels)) > 1 and len(cluster_labels) > num_clusters and cluster_count == num_clusters:
            score = silhouette_score(embedding, cluster_labels)
        else:
            score = -1  
        
        return score
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials= n_trials, timeout=1000, show_progress_bar= True)  # Adjust as needed

    # 5. Output the best hyperparameters
    best_params = study.best_params
    best_score = study.best_value



    print("Best Silhouette Score:", best_score)
    print("Best Hyperparameters:", best_params)


    # 7. Apply the best hyperparameters
    best_reducer = umap.UMAP(
        n_neighbors=best_params['n_neighbors'],
        min_dist=best_params['min_dist'],
        n_components=best_params['n_components'],
        metric='cosine',
        random_state=42,
        n_jobs=1,
    )

    best_embedding = best_reducer.fit_transform(embeddings_scaled)

    kmeans = KMeans(n_clusters=best_params['n_clusters'], random_state=42,  n_init = "auto")
    best_cluster_labels = kmeans.fit_predict(best_embedding)

    end = time.time()

    print(f"Clustering was completed in {round((end-start),2)}s.")

    cluster_record = {
            'n_components' : best_params['n_components'],
            'n_clusters' : best_params['n_clusters'],
            'silhouette_score' : best_score,
            'time' : round((end-start),2)
        }

    return best_cluster_labels, cluster_record


def update_document(document,best_cluster_labels):

    '''
        Update the sentece objects assigning cluster label of each sentence
    '''
    for obj, cluster_label in zip(document, best_cluster_labels):
        obj.set_cluster_label(cluster_label)

def prepare_clusters(document:list[Sentence], n_clusters : int)-> dict[int:list[Sentence]]:
    '''
    Return clustered organization of sentences as a dictionary
    '''
    clustered_document = dict()
    for label in range(n_clusters):
        cluster = list(filter(lambda sent:sent.cluster == label, document))
        sorted_cluster = sorted(cluster, key=lambda Sentence: Sentence.index)
        clustered_document[label] = sorted_cluster
    
    return clustered_document

def evaluate(X:np.array, y_pred: np.array):
   '''
   Internal validation of clustering without ground truth labels
   '''
   clustering_evaluator = ClusteringMetric(X=X, y_pred = y_pred)
   metrics_dict = {
      "DBI": None,   # Davies Bouldin Index
      "SI" : None,   # Silhouette Index
      "DBCVI": None,    # Density Based Clustering Validation Index
      "DI" : None,   # Dunn Index
      "CHI" : None, # Calinski Harabasz Index
   }
   evaluation_report = clustering_evaluator.get_metrics_by_dict(metrics_dict)
   return evaluation_report

## Main function to clustering pipeline

def k_means_cluster_document(document:list[Sentence], n_trials: int = 10) -> tuple[dict[int:list[Sentence]],dict]:
    '''
    Cluster given document using kmeans clusering with hyperparameter tuning.
    parameters:
        document: list of sentence objects
        n_trials: number of trails to be tried in hyperparameter tuning default 10
    Returns:
        clusters : dict
        clustering report: dict

    '''
    no_sentences = len(document)
    n = min(25, no_sentences//100)
    n_clusters_min = n - 5
    n_clusters_max = n + 5

    # prepare data for clsutering

    embedding_matrix = get_feature_matrix(document = document)

    # perform clustering with hyperparameter tuning

    best_cluster_labels, cluster_report = reduce_and_cluster(feature_mat = embedding_matrix, n_trials = n_trials, n_clusters_min = n_clusters_min, n_clusters_max = n_clusters_max)

    # update sentence structure
    update_document(document=document, best_cluster_labels= best_cluster_labels)

    clustering = prepare_clusters(document=document,n_clusters=cluster_report['n_clusters'])

    evaluation_report = evaluate(X = embedding_matrix, y_pred=best_cluster_labels)

    cluster_report.update(evaluation_report)

    return clustering, cluster_report