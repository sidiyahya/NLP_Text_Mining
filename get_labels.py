from sklearn import preprocessing
import numpy as np

def get_true_labels(data_labels):
    global true_labels
    if (type(data_labels[0]) == str):
        if (data_labels[0].isnumeric()):
            true_labels = [int(i) for i in data_labels.tolist()]
        else:
            le = preprocessing.LabelEncoder()
            le.fit(np.unique(data_labels))
            true_labels = list(le.transform(data_labels.tolist()))
    elif (type(data_labels[0]) == int):
        true_labels = data_labels.tolist()

    return true_labels