import pickle
import gensim

best_score = 0;

#come back and do better optimization

#no_n_below should be uint, ex: no_n_below = 3 or no_n_below = 5
#no_freq_above should be float [0,1], ex: no_freq_above = 0.5
#n_feats should be uint, ex: n_feats = 1024 or n_feats = 2048
def create_dictionary(tokenized_documents, n_feats, no_n_below = 3, no_freq_above = 0.5):
    id2word_dict = gensim.corpora.Dictionary(tokenized_documents)
    id2word_dict.filter_extremes(no_below = no_n_below, no_above = no_freq_above, keep_n = n_feats, keep_tokens = None) 
    return id2word_dict   
	
def corpus_tf(id2word_dict, tokenized_documents):
    return [id2word_dict.doc2bow(document) for document in tokenized_documents]

def loop_lda(tokenized_documents, 
                     tfcorpus, 
                     id2word_dict,
                     start, #suggest 2 or something
                     stop, # suggest 20 or similar
                     step,
					 ngram,
					 n_feats,
                     per_word_topics = False): #compute list of topics for each word
	topic_counts = []
	coherence_scores = []
	for n_topics in range (start, stop, step):
		lda_model = gensim.models.ldamodel.LdaModel(corpus = tfcorpus, num_topics = n_topics, id2word = id2word_dict, per_word_topics = per_word_topics)
		coherence_model = gensim.models.CoherenceModel(model = lda_model, texts = tokenized_documents, dictionary = id2word_dict, coherence = "c_v")
		coherence_score = coherence_model.get_coherence()
		coherence_scores.append(coherence_score)
		topic_counts.append(n_topics)
		print("coherence of %f with %i topics and %i features" % (coherence_score, n_topics, n_feats))
		if coherence_score > best_score:
			best_score = coherence_score
			bestname = "ldamodel_" + str(ngram) + "gram_" + str(n_feats) + "feats_" + str(n_topics) + "topics.pkl"
			bestpath = os.path.join("./models", bestname)
			with open(bestpath, 'wb') as f:
				python.dump(bestpath, f)
	return topic_counts, coherence_scores;
	
def loop_ntopics_lda(tokenized_documents, n_feats, start, stop, step, ngram):
	id2word_dict = create_dictionary(tokenized_documents, n_feats = n_feats)
	tfcorpus = corpus_tf(id2word_dict, tokenized_documents)
	topic_counts, coherence_scores = loop_lda(tokenized_documents, tfcorpus, id2word_dict, start, stop, step, ngram, n_feats)
	gc.collect()
	return topic_counts, coherence_scores

tokenized_path = "tokenized_documents.pkl"
print("opening %s" % str(tokenized_path))
with open(tokenized_path, 'rb') as f:
	tokenized_documents = pickle.load(f)

print(len(tokenized_documents))

start = 2
stop = 20
step = 1
temp = 10

n_feats = 256
n_feats = 512
n_feats = 1024

id2word_dict = create_dictionary(tokenized_documents, n_feats = n_feats)
tfcorpus = corpus_tf(id2word_dict, tokenized_documents)

lda_model = gensim.models.ldamodel.LdaModel(corpus = tfcorpus, num_topics = temp, id2word = id2word_dict, per_word_topics = False)

coherence_model = gensim.models.CoherenceModel(model = lda_model, texts = tokenized_documents, dictionary = id2word_dict, coherence = "c_v")
coherence_score = coherence_model.get_coherence()
print("coherence score of %d" % (coherence_score))


