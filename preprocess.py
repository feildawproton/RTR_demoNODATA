import os
import docx2txt
import gensim
import gc
import pickle

def get_documents(folder = "./data"):
	documents = {"doc_name" : [], "text" : [] }
	for file in os.listdir(folder):
		if file.endswith(".docx"):
			documents["doc_name"].append(str(file))
			text = docx2txt.process(os.path.join(folder, file))
			documents["text"].append(text)
	return documents
	
def make_bigram(tokenized_data, min_count = 5, threshold = 100):
	bigram_phrases = gensim.models.Phrases(tokenized_data, min_count = min_count, threshold = threshold)
	#after Phrases a Phraser is faster to access
	bigram = gensim.models.phrases.Phraser(bigram_phrases)
	gc.collect()
	return bigram
	
def make_trigram(tokenized_data, min_count = 5, threshold = 100):
	bigram_phrases = gensim.models.Phrases(tokenized_data, min_count = min_count, threshold = threshold)
	trigram_phrases = gensim.models.Phrases(bigram_phrases[tokenized_data], threshold = 100)
	#after Phrases a Phraser is faster to access
	trigram = gensim.models.phrases.Phraser(trigram_phrases)
	gc.collect()
	return trigram
	
print("getting documents")
documents = get_documents();

print("saving the document dictionary of lengths %i and %i for use later" % (len(documents["doc_name"]), len(documents["text"])))
with open("documents_dict.pkl", 'wb') as f:
	pickle.dump(documents, f)
	
print("Remove stopwords, stem, and tokenize %i documents" % (len(documents["text"])))
tokenized_documents =  gensim.parsing.preprocessing.preprocess_documents(documents["text"])

print("saving tokensized documents of length %i for use later" % (len(tokenized_documents)))
with open("tokenized_documents.pkl", 'wb') as f:
	pickle.dump(tokenized_documents, f)
	
print("creating bigram documents")
bigram_documents = make_bigram(tokenized_documents)
print("saving bigram_documents")
with open("bigram_documents.pkl", 'wb') as f:
	pickle.dump(bigram_documents, f)
	
print("creating trigram documents")
trigram_documents = make_trigram(tokenized_documents)
print("saving trigram documents")
with open("trigram_documents.pkl", 'wb') as f:
	pickle.dump(trigram_documents, f)


	
