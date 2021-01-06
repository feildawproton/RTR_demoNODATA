# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 10:01:55 2020

@author: alexander.feild
"""

import os
import pickle
import gensim

start_feats_log2 = 7
stop_feats_log2 = 14
start_topics_log2 = 1
stop_topics_log2 = 7

def get_models():

    print("Opening models")
    
    model_data = {"corpora" : [], "dictionaries" : [], "lda_models" : [], "similarity_indices" : [], "model_names" : []}
    
    feats_iter_log2 = start_feats_log2
    while(feats_iter_log2 <= stop_feats_log2):
        feats = 2**feats_iter_log2
        
        topics_iter_log2 = start_topics_log2
        while(topics_iter_log2 <= stop_topics_log2):
            n_topics = 2**topics_iter_log2
            
            name = str(feats) + "feats_" + str(n_topics) + "topics_Corpus_.pkl"
            path = os.path.join("models", name)
            with open(path, 'rb') as f:
                model_data["corpora"].append(pickle.load(f))
            
            name = str(feats) + "feats_" + str(n_topics) + "topics_Dictionary_.pkl"
            path = os.path.join("models", name)
            with open(path, 'rb') as f:
                model_data["dictionaries"].append(pickle.load(f))
            
            name = str(feats) + "feats_" + str(n_topics) + "topics_LDAmodel_.pkl"
            path = os.path.join("models", name)
            with open(path, 'rb') as f:
                model_data["lda_models"].append(pickle.load(f))
                
            model_data["model_names"].append(str(feats) + "feats_" + str(n_topics) + "topics")
            
            topics_iter_log2 += 1
            
        feats_iter_log2 += 1
    
    print("Creating similarity indices")
    for i, model in enumerate(model_data["lda_models"]):
        model_data["similarity_indices"].append(gensim.similarities.MatrixSimilarity(model[model_data["corpora"][i]]))
    print("done creating similarity indices")
    
    return model_data

model_data = get_models()

data_path = os.path.join("workingdata", "model_data.pkl")
with open(data_path, 'wb') as f:
    data = pickle.dump(model_data, f)
    


