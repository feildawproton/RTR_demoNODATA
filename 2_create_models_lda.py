# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 13:46:50 2020

@author: alexander.feild
"""
import os
import pickle
import gensim

def create_dictionary(tokenized_documents, n_feats, no_n_below = 3, no_freq_above = 0.5):
    id2word_dict = gensim.corpora.Dictionary(tokenized_documents)
    print("prior dictionary len %i" % len(id2word_dict))
    id2word_dict.filter_extremes(no_below = no_n_below, no_above = no_freq_above, keep_n = n_feats, keep_tokens = None)
    print("current dictionary len %i" % len(id2word_dict))
    return id2word_dict

def save_model(model, dictionary, corpus, n_feats, n_topics):
    name =  str(n_feats) + "feats_" + str(n_topics) + "topics_" + "LDAmodel_" + ".pkl"
    path = os.path.join("models", name)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    
    name =  str(n_feats) + "feats_" + str(n_topics) + "topics_" + "Dictionary_" + ".pkl"
    path = os.path.join("models", name)
    with open(path, 'wb') as f:
        pickle.dump(dictionary, f)
        
    name =  str(n_feats) + "feats_" + str(n_topics) + "topics_" + "Corpus_" + ".pkl"
    path = os.path.join("models", name)
    with open(path, 'wb') as f:
        pickle.dump(corpus, f)
    
start_feats_log2 = 7
stop_feats_log2 = 14
start_topics_log2 = 1
stop_topics_log2 = 7

data_path = os.path.join("workingdata", "dataset.pkl")
with open(data_path, 'rb') as f:
    data = pickle.load(f)
    
data_path = os.path.join("workingdata", "trigram_model.pkl")
with open(data_path, 'rb') as f:
    trigram_model = pickle.load(f)
    
print("opened data with %i preprocessed documents" % (len(data["trigram_texts"])))


feats_iterator_log2 = start_feats_log2
while(feats_iterator_log2 <= stop_feats_log2):
    n_feats = 2**feats_iterator_log2
    print("using %i features" % (n_feats))
    
    print("creating dictionary")
    id2word_dict = create_dictionary(data["trigram_texts"], n_feats)
    print("creating corpus")
    tfcorpus = [id2word_dict.doc2bow(doc) for doc in data["trigram_texts"]]

    best = 0
    n_topics_log2 = start_topics_log2
    while(n_topics_log2 <= stop_topics_log2):
        n_topics = 2**n_topics_log2
        print("using %i topics" % (n_topics))
        
        print("creating lda model")
        lda_model = gensim.models.ldamodel.LdaModel(corpus = tfcorpus, num_topics = n_topics, id2word = id2word_dict, per_word_topics = False)

        save_model(lda_model, id2word_dict, tfcorpus, n_feats, n_topics)
        
        n_topics_log2 += 1
    
    feats_iterator_log2 += 1




