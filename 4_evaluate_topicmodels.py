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
    
def get_models():

    print("Opening model data")
    
    models_path = os.path.join("workingdata", "model_data.pkl")
    with open(models_path, "rb") as f:
        model_data = pickle.load(f)
    return model_data
        
def evaluate_modelanddoc(lda_model, text, trigram_model, dictionary, similarity_index, name, Names):
    snippet_len = int(len(text)/16)
    snippet = text[0:snippet_len]
    tokenized_snippet = gensim.parsing.preprocess_string(snippet)
    trigram_snippet = trigram_model[tokenized_snippet]
    vector_snippet = dictionary.doc2bow(trigram_snippet)
    topic_snippet = lda_model[vector_snippet]
    similarities = similarity_index[topic_snippet]
    ranked_indices = sorted(enumerate(similarities), key = lambda item: -item[1])
    
    score = 0
    for i, index in enumerate(ranked_indices):
        if Names[index[0]] == name:
            score += float(1 / (i+1))
    return score

def evaluate_model(lda_model, doc_data, trigram_model, dictionary, similarity_index):
    score = 0
    for i, text in enumerate(doc_data["texts"]):
        score += evaluate_modelanddoc(lda_model, text, trigram_model, dictionary, similarity_index, doc_data["names"][i], doc_data["names"])
    return score


model_data = get_models()

path = os.path.join("workingdata", "dataset.pkl")
with open(path, 'rb') as f:
    doc_data = pickle.load(f)
    
path = os.path.join("workingdata", "trigram_model.pkl")
with open(path, 'rb') as f:
    trigram_model = pickle.load(f)

'''
query = ""
while exit_check(query) == False:
    query = input("Query: ")
    similarities(query, trigram_model, model_data, doc_data)
'''

scores = []
scoresum = 0
for i, model in enumerate(model_data["lda_models"]):
    score = evaluate_model(model, doc_data, trigram_model, model_data["dictionaries"][i], model_data["similarity_indices"][i])
    print("model %i with score %f" % (i, score))
    scoresum += score
    scores.append(score)
    
model_weightedscores = {"model_names" : [], "weighted_scores" : []}
for score in scores:
    model_weightedscores["weighted_scores"].append(score/scoresum)

model_weightedscores["model_names"] = model_data["model_names"]

print(model_weightedscores)

path = os.path.join("workingdata", "model_weightedscores")
with open(path, 'wb') as f:
    pickle.dump(model_weightedscores, f)


    
    
    
    
    
    
    