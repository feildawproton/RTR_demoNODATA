# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 09:34:43 2020

@author: alexander.feild
"""
import os
import pickle
import gensim

start_feats_log2 = 7
stop_feats_log2 = 14
start_topics_log2 = 1
stop_topics_log2 = 7

def exit_check(query):
    if query == "exit" or query == "Exit" or query == "EXIT" or query == "ExIt" or query == "eXiT":
        return True
    else:
        return False

def get_topic_query(query, trigram_model, dictionary, lda_model):
    #1.) tokenize query
    #2.) trigram query
    #3.) vector query in corpus terms
    #4.) finally topic based query
    #1.)
    tokenized_query = gensim.parsing.preprocess_string(query)
    #2.)
    trigram_query = trigram_model[tokenized_query]
    #3.)
    vector_query = dictionary.doc2bow(trigram_query)
    #4)
    return lda_model[vector_query]

def evaluate_model_qeury(lda_model, text_query, trigram_model, dictionary, similarity_index):
    tokenized_qeury = gensim.parsing.preprocess_string(text_query)
    trigram_qeury = trigram_model[tokenized_qeury]
    vector_qeury = dictionary.doc2bow(trigram_qeury)
    topic_query = lda_model[vector_qeury]
    similarities = similarity_index[topic_query]
    #return sorted(enumerate(similarities), key = lambda item: -item[1])
    return similarities

path = os.path.join("workingdata", "model_data.pkl")
with open(path, 'rb') as f:
    model_data = pickle.load(f)

path = os.path.join("workingdata", "dataset.pkl")
with open(path, 'rb') as f:
    doc_data = pickle.load(f)
    
path = os.path.join("workingdata", "trigram_model.pkl")
with open(path, 'rb') as f:
    trigram_model = pickle.load(f)
    
path = os.path.join("workingdata", "model_weightedscores")
with open(path, 'rb') as f:
    weighted_scores = pickle.load(f)

query = ""
while exit_check(query) == False:
    query = input("Query: ")
    sum_similarities = [0]*len(doc_data["names"])
    sum_weighted_similarities = [0]*len(doc_data["names"])
    for i, model in enumerate(model_data["lda_models"]):

        similarities = evaluate_model_qeury(model, query, trigram_model, model_data["dictionaries"][i], model_data["similarity_indices"][i])
        sum_similarities = [(sum_similarities[j] + similarity) for j, similarity in enumerate(similarities)]
        
        weight = weighted_scores["weighted_scores"][i]
        sum_weighted_similarities = [(sum_weighted_similarities[j] + similarity*weight) for j, similarity in enumerate(similarities)]
        
    #print(sum_similarities)
    sorted_indices = sorted(enumerate(sum_similarities), key = lambda item: -item[1])
    print("unweighted results")
    print(doc_data["names"][sorted_indices[0][0]])
    print(doc_data["names"][sorted_indices[1][0]])
    print(doc_data["names"][sorted_indices[2][0]])
    
    
    sorted_weighted = sorted(enumerate(sum_weighted_similarities), key = lambda item: -item[1])
    print("weighted results")
    print(doc_data["names"][sorted_weighted[0][0]])
    print(doc_data["names"][sorted_weighted[1][0]])
    print(doc_data["names"][sorted_weighted[2][0]])
    
    




    
    
    
    
    
    
    