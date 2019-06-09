from data_preprocessing import DataPreprocessing
from model import Skipgram
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np



class Word2vec:
    def __init__(self,
                 batch_size = 100,
                 context_size = 5,
                 embedding_dim = 300, 
                 epoch_num = 5000,
                 num_neg_words = 5,
                 lr = 0.025):
        super(Word2vec, self).__init__()
        
        self.batch_size = batch_size
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.epoch_num = epoch_num
        self.num_neg_words = num_neg_words
        self.lr = lr
      
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.model.cuda()
        

    def fit(self, input_file, use_pretrained_model = False, file_with_model = None, draw_losses = False):
        
        self.prep_data = DataPreprocessing(input_file)
        self.vocab_size = len(self.prep_data.vocab)
        
        self.model = Skipgram(self.vocab_size, self.embedding_dim)
        
        if use_pretrained_model:
            self.load_model(file_with_model)
        
        self.optimizer = optim.SGD(self.model.parameters(), lr = self.lr)
        losses = []
        
        for epoch in tqdm(range(self.epoch_num)):
            positive_pairs = self.prep_data.get_batch_pairs(self.batch_size, self.context_size)
           
            pos_centers = Variable(torch.LongTensor([pair[0] for pair in positive_pairs]))
            pos_contexts = Variable(torch.LongTensor([pair[1] for pair in positive_pairs]))
            
            negative_neighbors = self.prep_data.get_negative_pairs(positive_pairs, self.num_neg_words)
            negative_neighbors = Variable(torch.LongTensor(negative_neighbors))
            
            if self.use_cuda:
                pos_centers = pos_centers.cuda()
                pos_contexts = pos_contexts.cuda()
                negative_neighbors = negative_neighbors.cuda()
            
            self.optimizer.zero_grad()
            loss = self.model.forward(pos_centers, pos_contexts, negative_neighbors)
            loss.backward()
            
            self.optimizer.step()
            
            losses.append(loss.data)
        
        print('Optimisation completed')
        self.save_embeddings('final_embed')
        if draw_losses:
            plt.figure(figsize=(15,10))
            plt.plot(losses)
            plt.show()
        
        
    def get_similar(self, target_word, n_similar):
        target_word_emb = self.get_embedding(target_word)
        norm_target_word = np.linalg.norm(target_word_emb)
        values = []
        for word in self.prep_data.vocab:
            if word != target_word:
                word_emb = self.get_embedding(word)
                norm_word = np.linalg.norm(word_emb)
                dot_product = np.dot(target_word_emb, word_emb)
                value = dot_product / (norm_target_word * norm_word)
                values.append((word, value))

        return [(w,v) for w, v in sorted(values, key = lambda x: x[1], reverse = True)][:n_similar]
        
    
    def save_model(self, output_file):
        torch.save(self.model.state_dict(), output_file)
    
    def load_model(self, input_file):
        self.model.load_state_dict(torch.load(input_file))
                
    def get_embedding(self, word):
        word_ind = self.prep_data.word2id[word]
        return self.model.get_embedding(word_ind)
    
    def save_embeddings(self, output_file):
        embeddings = self.model.u_embeddings.weight.data.numpy()
        
        with open(output_file, 'w') as f:
            for word_id, word in self.prep_data.id2word.items():
                emb = embeddings[word_id]
                f.write('({}, {})\n'.format(word, str(emb)))
    
    def get_words(self):
        return self.prep_data.vocab