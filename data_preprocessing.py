from collections import deque
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer 
import numpy as np
import nltk
stop_words = set(stopwords.words('english'))

class DataPreprocessing:
    def __init__(self, input_file):
        self.input_file = input_file
        self.vocab = None
        self.word_freq = dict()
        self.word2id = dict()
        self.id2word = dict()
        self.word_pairs = deque()
        self.prepared_documents = []
        self.document_count = 0
        self.sample_table = []
        self.doc_amount = 0
        self.build_vocab()
        self.make_sample_table()

        
    def build_vocab(self):
        word_freq = dict()
        lemma = WordNetLemmatizer() 
        w_id = 0
        w = []
        prep_documents = []
        for document in self.input_file:
            words = word_tokenize(' '.join(document).lower())
    
            for word in words:
                if word.isalpha() and word not in stop_words:
                    word = lemma.lemmatize(word)
                    w.append(word)
                    try: 
                        word_freq[word] += 1
                    except: 
                        word_freq[word] = 1                    
        
            prep_documents.append(w)
            self.doc_amount += 1
            w = []
        
        max_freq = max(word_freq.values())
        high_bound = max_freq - 300        
        w_id = 0
        for word, freq in word_freq.items():
            if freq > 20 and freq < high_bound and len(word) > 2:
                self.word2id[word] = w_id
                self.id2word[w_id] = word
                self.word_freq[w_id] = freq
                w_id+=1
   
        self.prepared_documents = [[word for word in document if word in self.word2id] for document in prep_documents]
        self.vocab = list(self.word2id.keys())
    
    def get_batch_pairs(self, batch_size, context_size):
        
        while len(self.word_pairs) < batch_size:   
            doc_num = self.document_count % self.doc_amount
            doc_words_ids = [self.word2id[word] for word in self.prepared_documents[doc_num]]
            
            self.document_count+=1
            
            for ind_center, center_word_id in enumerate(doc_words_ids):
                for ind_context, context_word_id in enumerate(doc_words_ids[max(ind_center - context_size, 0):ind_center+context_size+1]):
                    assert center_word_id < len(self.word2id)
                    assert context_word_id < len(self.word2id)
                    if ind_center == ind_context:
                        continue
                    self.word_pairs.append((center_word_id, context_word_id))
        batch_pairs = []
        
        for _ in range(batch_size):
            batch_pairs.append(self.word_pairs.popleft())
            
        return batch_pairs
    
    
    def make_sample_table(self):
        table_size = 1e8
        numerator = np.array(list(self.word_freq.values())) ** 0.75
        denominator = sum(numerator)
        ratio = numerator / denominator
        count = np.round(ratio * table_size)
        for word_ind, count in enumerate(count):
            self.sample_table += [word_ind] * int(count)
            
        self.sample_table = np.array(self.sample_table)

        
    def get_negative_pairs(self, pos_words_pair, k):
        
        neg_pairs = np.random.choice(self.sample_table, size=(k, len(pos_words_pair))).tolist()
        return neg_pairs
        
        
    def get_words(self):
        return self.vocab
    
    