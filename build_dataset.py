import ast
import numpy as np
import pandas as pd
from sklearn import preprocessing


def get_datasets_built(dataset_name):
    data = pd.DataFrame()
    classes_count = pd.DataFrame()
    stemmed = False
    if(dataset_name in ['r8', 'r52']):
        ##------------BUILDING THE DATASET
        X_train = pd.read_csv('datasets/'+dataset_name+'-train-all-terms.txt', sep="\t", header=None)
        X_test = pd.read_csv('datasets/'+dataset_name+'-test-all-terms.txt', sep="\t", header=None)
        data = pd.concat([X_train, X_test], ignore_index=True)
        data.columns = ["class", "text"]


    elif(dataset_name=='r40'):
        # ---------------STARTING WITH R40----------------------------
        ##------------BUILDING THE DATASET
        X_train = pd.read_csv('datasets/r40_texts.txt', header=None)
        X_labels = pd.read_csv('datasets/r40_labels.txt', header=None)
        data = pd.concat([X_labels, X_train], axis=1, ignore_index=True)
        data.columns = ["class", "text"]


    elif(dataset_name in ['classic3', 'classic4']):
        # ---------------Dataset CLASSIC4 & CLASSIC3----------------------------
        ##------------BUILDING THE DATASET
        list_ = []
        with open("datasets/"+dataset_name+".json", 'r+') as g:
            for x in g:
                list_.append(ast.literal_eval(x))

        data = pd.DataFrame(list_)
        data.columns = ["text", "class"]

    elif(dataset_name=='webkb'):
        # ---------------DATASET WEBKB STEMMED----------------------------
        ##------------BUILDING THE DATASET
        stemmed = True
        X_train = pd.read_csv('datasets/webkb-train-stemmed.txt', sep="\t", header=None)
        X_test = pd.read_csv('datasets/webkb-test-stemmed.txt', sep="\t", header=None)
        data = pd.concat([X_test, X_train], ignore_index=True)
        data.columns = ["class", "text"]

    classes_count = data.groupby('class').count().sort_values(by=['text'], ascending=False)

    return data, classes_count, stemmed


