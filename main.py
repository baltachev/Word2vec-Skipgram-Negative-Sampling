from word2vec import Word2vec
from sklearn.datasets import fetch_20newsgroups

# dataset
dataset = fetch_20newsgroups(
    subset='train', 
    categories=['comp.sys.mac.hardware', 'soc.religion.christian', 'rec.sport.hockey'])
data = [dataset.data]

w2v = Word2vec(50)

# train model
w2v.fit(data)

# get top k similar word
print(w2v.get_similar('jesus', 3))
print(w2v.get_similar('computer', 3))

# get vocabulary (10 first words)
print(w2v.get_words()[:10])

# save embeddings 
w2v.save_embeddings('embeddings')