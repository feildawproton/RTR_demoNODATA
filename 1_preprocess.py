# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 12:14:57 2020

@author: alexander.feild
"""

import os
#import docx2txt
import gensim
import pickle

from sklearn.datasets import fetch_20newsgroups

'''
def fetch_docx(folder = "./PWSdata"):
    print("loading %s data" % folder)
    texts = []
    names = []
    for file in os.listdir(folder):
        if file.endswith(".docx"):
            texts.append(docx2txt.process(os.path.join(folder, file)))
            names.append(str(file))
    return texts, names;
    '''

def fetch_data(subset = "train"):
    print("downloading %s data." % subset)
    #remove = a tuple for removing a subset of ("headers", "footers", and "quotes").  prevents classifiers from overfitting metadata
    #download_if_missing = on by default
    Dataset = fetch_20newsgroups(subset = subset, shuffle = True, random_state = 1, remove = ("headers", "footers", "quotes"))
    #dataset.target_names #if you want the names of the target categories
    print("done loading %s data" % subset)
    return Dataset.data, Dataset.filenames

'''
def fetch_docx(folder = "./PWSdata"):
    print("loading %s data" % folder)
    docdata = {"names" : [], "texts": []}
    for file in os.listdir(folder):
        if file.endswith(".docx"):
            text = docx2txt.process(os.path.join(folder, file))
            docdata["texts"].append(text)
            docdata["names"].append(str(file))
    return docdata
'''

def split_sentences(docs):
    doc_sentences = [gensim.summarization.textcleaner.clean_text_by_sentences(text) for text in docs]
    sentences = []
    for doc in doc_sentences:
        for sentence in doc:
            sentences.append(sentence.text)
    return sentences

#Data, FileNames = fetch_fromfolder()
#print("fetching PWS documents")
print("fetching the 20 newsgroups data")
texts, names = fetch_data()
print("loaded %i documents" % (len(names)))

print("splitting documents into sentences and combining them")
sentences = split_sentences(texts)
print("Done splitting sentences with %i sentences" % (len(sentences)))
print(sentences[9])


print("Removing stopwords, stemming, and tokenizing sentences")
tokenized_sentences = gensim.parsing.preprocessing.preprocess_documents(sentences)
print("done preprocessing %i sentences" % (len(tokenized_sentences)))     

print("making bigram model")
#the reason I broke dataset into sentences is because this funtion takes sentences
bigram = gensim.models.Phrases(tokenized_sentences, min_count = 3)
bigram_model = gensim.models.phrases.Phraser(bigram)
print("done creating bigram model. Creating trigram model")
trigram = gensim.models.Phrases(bigram[tokenized_sentences])
trigram_model = gensim.models.phrases.Phraser(trigram)
print("done creating trigram model")

print("Removing stopwords, stemming, and tokenizing documents")
tokenized_documents = gensim.parsing.preprocessing.preprocess_documents(texts)
print("done preprocessing %i documents" % (len(tokenized_documents)))

trigram_texts = [trigram[doc] for doc in tokenized_documents]

dataset = {"names" : names, "texts" : texts, "trigram_texts" : trigram_texts}

dataset_path = os.path.join("workingdata", "dataset.pkl")
print("saving document data (including document trigrams)")
with open(dataset_path, 'wb') as f:
    pickle.dump(dataset, f)

trigram_path = os.path.join("workingdata", "trigram_model.pkl")
print("saving trigram model.  don't need to save bigram model because it is a part of the trigram model")
with open(trigram_path, 'wb') as f:
    pickle.dump(trigram_model, f)





