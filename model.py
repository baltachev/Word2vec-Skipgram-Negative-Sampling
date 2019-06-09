import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class Skipgram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Skipgram, self).__init__()
        self.embedding_dim = embedding_dim
        self.u_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.v_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        initrange = 0.5 / self.embedding_dim
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0,0)
   
        
    def forward(self, input_words, context_words, neg_words):
        
        emb_u = self.u_embeddings(input_words)
        emb_v = self.v_embeddings(context_words)
        negative_value = 0
        positive_value = F.logsigmoid(torch.sum(torch.diag(torch.mm(emb_u, emb_v.t()))))
  
        for neg in neg_words:
            neg_emb = self.v_embeddings(neg)
            neg_value = torch.sum(torch.diag(torch.mm(emb_u, neg_emb.t())))
            negative_value += F.logsigmoid(-neg_value)

        loss = -1 * (positive_value + negative_value)
        
        return loss
    
                
    def get_embedding(self, word):
        embedding = self.u_embeddings.weight.data.numpy()
        
        return embedding[word]
        