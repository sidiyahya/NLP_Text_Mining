# NLP_Text_Mining
## Files Description :
- datasets : contains all datasets
- results : algorithms output
- notebooks : notebooks for test
- build_dataset : to import/build all of the datasets
- execute_coclustering : excute coclustering with ["CoclustMod", "CoclustInfo", "CoclustModFuzzy"] and return the evaluation result (nmi, ari, accuracy) 
with Term Clustering 
- get_labels : get the true labels (row) to compare with coclustering results : encoding labels if necessary like "earn" ==> 1
- nlp_preprocessing : contains two methods for preprocessing 
- similarities : algorithms for columns clustering evaluation , it returns the accuracy and L wortest results for FP and FN
- exploration.ipynb : a notebooks that shows graphs/plots for data exploration perpouse
- main.ipynb : the main file to put all the previous methods/codes together and execute all the pipeline