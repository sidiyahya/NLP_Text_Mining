import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_pair_words(sim_mat):
    pair_words = sim_mat.where(np.triu(np.ones(sim_mat.shape), 1).astype(np.bool))
    pair_words = pair_words.stack().reset_index()
    pair_words.columns = ['Row', 'Column', 'Value']
    pair_words = pair_words.sort_values('Value')
    return pair_words


def compute_column_metrics(column_labels, word_vectors, df_vocab, alphas=None):
    if alphas is None:
        alphas = [0.85, 0.7, 0.45]
    accs = []
    all_fp_words = []
    all_fn_words = []

    sim_matrix = similarity_matrix(word_vectors, df_vocab)

    labels = np.unique(column_labels)
    for alpha in alphas:
        tp = tn = fp = fn = 0
        for l in labels:
            ind = np.where(column_labels == l)[0]

            # Get the similarity matrix of the cluster
            sim_matrix_label = sim_matrix.iloc[ind,].iloc[:, ind]

            # Get pair words names and score
            fp_pair_words = get_pair_words(sim_matrix_label)
            fp_pair_words = pd.concat([fp_pair_words[:200], fp_pair_words[-200:]])
            all_fp_words.append((alpha, l, fp_pair_words))

            # Compute metrics
            tocompare = np.copy(sim_matrix_label)
            tocompare[np.tril_indices(tocompare.shape[0], 0)] = np.nan
            tp += (tocompare >= alpha).sum()
            fp += (tocompare < alpha).sum()

        # Filter the sim_matrix by keeping values of words being not in the same cluster
        column_labels = np.array(column_labels)
        matrix_filter = np.matrix(list(map(lambda x: x == column_labels, column_labels)))
        sim_matrix_diff = sim_matrix.copy()
        # It will raise some warnings when we compare with alpha below
        sim_matrix_diff[matrix_filter] = None

        # Compute metrics
        tocompare = np.copy(sim_matrix_diff)
        tocompare[np.tril_indices(tocompare.shape[0], 0)] = np.nan
        tn += (tocompare < alpha).sum()
        fn += (tocompare >= alpha).sum()

        # Get pair words names and score
        fn_pair_words = get_pair_words(sim_matrix_diff)
        fn_pair_words = pd.concat([fn_pair_words[:200], fn_pair_words[-200:]])
        all_fn_words.append((alpha, fn_pair_words))

        acc = (tp + tn) / (tp + tn + fp + fn)
        accs.append(acc)

    return accs, all_fp_words, all_fn_words


def similarity_matrix(word_vectors, df_vocab):
    sim_mat = pd.DataFrame(cosine_similarity(word_vectors), columns=df_vocab, index=df_vocab)
    print(sim_mat.shape)

    return sim_mat