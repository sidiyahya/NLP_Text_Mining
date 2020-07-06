from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari
from coclust.coclustering import CoclustInfo, CoclustMod, CoclustSpecMod
from coclust.evaluation.external import accuracy
import numpy as np

from scipy.io import loadmat


def execute_coclustering(tf_idf, method, n_clusters, return_pred_rows=True, max_iteration=300):
    global model
    print("---executing ",method)
    if(method=="CoclustInfo"):
        model = CoclustInfo(n_row_clusters=n_clusters, n_col_clusters=n_clusters, n_init=10, max_iter=max_iteration)
    elif(method=="CoclustMod"):
        model = CoclustMod(n_clusters=n_clusters, n_init=10, max_iter=max_iteration)
    elif(method=="CoclustModFuzzy"):
        model = CoclustSpecMod(n_clusters=n_clusters, n_init=10, max_iter=max_iteration)
    model.fit(tf_idf)
    pred_row_labels = model.row_labels_
    pred_column_labels = model.column_labels_
    if(return_pred_rows):
        return pred_row_labels
    else:
        return pred_column_labels

#%%

# Evaluate the results
def clustering_quality(true_row_labels, predicted_row_labels):
    nmi_ = nmi(true_row_labels, predicted_row_labels)
    ari_ = ari(true_row_labels, predicted_row_labels)
    acc_ = accuracy(true_row_labels, predicted_row_labels)
    print("NMI : {}\nARI : {}\nAccuracy : {}".format(nmi_, ari_, acc_))
    return nmi_, ari_, acc_

#%%

def execute_clustering_evaluation(raw_data, true_labels, row_labels=True,use_words_thresh=True, max_iteration=300):
    global tfidf_vectorizer
    clustering_eval = []
    n_labels = len(np.unique(true_labels))
    if(use_words_thresh):
        tfidf_vectorizer = TfidfVectorizer(use_idf=True, max_features=5000, max_df=0.7, min_df=0.001)
    elif(not use_words_thresh):
        tfidf_vectorizer = TfidfVectorizer(use_idf=True, max_features=5000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(raw_data)
    colustering_methods = ["CoclustMod", "CoclustInfo", "CoclustModFuzzy"]
    for algo in colustering_methods:
        pred_labels = execute_coclustering(tfidf_matrix, algo, n_labels, return_pred_rows=row_labels, max_iteration=max_iteration)
        nmi_, ari_, acc_ = clustering_quality(true_labels, pred_labels)
        clustering_eval += [[algo, nmi_, ari_, acc_]]
    return clustering_eval