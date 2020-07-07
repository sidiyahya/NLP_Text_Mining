import numpy as np
def similarity_matrix(word_2vec_model, vocabulary):
    similarity_mat = np.zeros((len(vocabulary), len(vocabulary)))
    for word in range(len(vocabulary)):
        for to_compare in range((vocabulary)):
            similarity_mat[word][to_compare] = word_2vec_model.similarity(vocabulary[word], vocabulary[to_compare])

    return similarity_mat

def execute_column_evaluations(coclust_column_labels, word_2vec_model, vocabulary, thresh=None, L=100):
    if thresh is None:
        thresh = [0.85, 0.7, 0.45]
    simit_matrix = similarity_matrix(word_2vec_model, vocabulary)
    accuracy_list = []
    fp_list = []
    fn_list = []
    labels = np.unique(coclust_column_labels)
    for alpha in thresh:
        TP = 0
        FN = 0
        FP = 0
        TN = 0
        for label in labels:
            same_cluster_obs = np.where(coclust_column_labels == label)[0]

            simit_matrix_obs = simit_matrix.iloc[same_cluster_obs, :].iloc[:, same_cluster_obs]

            lower_triangle = simit_matrix_obs.where(np.tril(np.ones(simit_matrix_obs.shape), -1).astype(bool)).stack().sort_values()
            TP += len(lower_triangle[lower_triangle[:] >= alpha])
            fp_values = lower_triangle[lower_triangle[:] < alpha]
            FP += len(fp_values)
            fp_dict = list(fp_values[:L].to_dict().items())
            """fp_dictlist = []
            for key, value in fp_dict.items():
                temp = [key, value]
                fp_dictlist.append(temp)"""
            fp_list += [[label, alpha, fp_dict]]

        for vocab in vocabulary:
            label = coclust_column_labels[vocab]
            same_cluster_obs = np.where(coclust_column_labels == label)[0]
            diffetrent_cluster_obs = simit_matrix.iloc[~same_cluster_obs, :].iloc[:, vocab]
            TN += len(diffetrent_cluster_obs[diffetrent_cluster_obs[:] < alpha])
            fn_dict = list(diffetrent_cluster_obs[diffetrent_cluster_obs[:] >= alpha][:L].to_dict().items())
            FN += len(fn_dict)


        fn_list += [[alpha, fn_dict]]

        accuracy = (TP + TN) / (TP + TN + FN + FP)
        accuracy_list.append(accuracy)
    return accuracy_list, fp_list, fn_list

