# dataloader.py
import os
import os.path as osp
import numpy as np
import pandas as pd
import re
import argparse

from gensim.models.keyedvectors import KeyedVectors

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class Vocabulary(object):  #??
    def __init__(self):
        self.word2idx = None
        self.idx2word = None
    
    def build_vocab(self, all_tokens):
        unique_tokens = list(set([word for word in all_tokens]))
        unique_tokens = ['<pad>', '<unk>'] + unique_tokens
        
        self.word2idx = {word:idx for idx, word in enumerate(unique_tokens)}
        self.idx2word = {idx:word for idx, word in enumerate(unique_tokens)}
    
    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]
    
    def __len__(self):
        return len(self.word2idx)
        

class CustomDataset(Dataset):
    def __init__(self, args):
        super(Dataset, self).__init__()
        
        self.args = args
        self.vocab = Vocabulary()
        self._build_vocab()
        
        print('>>> Dataset loading...')
        self.dataset = self.load_data()
        
        print('>>> Word2Vec loading...')
        word2vec = KeyedVectors.load_word2vec_format(r"C:\Users\Simon\ongoing_projects\SSRC_collaboration\source\GoogleNews-vectors-negative300.bin", binary=True)
        print('>>> Word2Vec loaded')
        self.pretrained_embeddings = self.build_embeddings(word2vec)

    def load_data(self):  # self.args
        """
        Load Data
        """
        data = []
        with open(osp.join(self.args.path, 'rt-polarity.pos.txt'), 'r', encoding="ISO-8859-1") as f:
            pos = f.readlines()
            for i in pos:
                tokens = self.clean_str(i)
                data.append(([self.vocab(t) for t in tokens], 1)) # (token_lst, label)
        
        with open(osp.join(self.args.path, 'rt-polarity.neg.txt'), 'r', encoding="ISO-8859-1") as f:
            neg = f.readlines()
            for i in neg:
                tokens = self.clean_str(i)
                data.append(([self.vocab(t) for t in tokens], 0)) 
        return data
    
    def clean_str(self, string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Every dataset is lower cased except for TREC 
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
        string = re.sub(r"\'s", " \'s", string) 
        string = re.sub(r"\'ve", " \'ve", string) 
        string = re.sub(r"n\'t", " n\'t", string) 
        string = re.sub(r"\'re", " \'re", string) 
        string = re.sub(r"\'d", " \'d", string) 
        string = re.sub(r"\'ll", " \'ll", string) 
        string = re.sub(r",", " , ", string) 
        string = re.sub(r"!", " ! ", string) 
        string = re.sub(r"\(", " \( ", string) 
        string = re.sub(r"\)", " \) ", string) 
        string = re.sub(r"\?", " \? ", string) 
        string = re.sub(r"\s{2,}", " ", string)
        string = string.strip().lower()
        return string.split()   # token list

    def _build_vocab(self):
        all_tokens = []
        with open(osp.join(self.args.path, 'rt-polarity.pos.txt'), 'r', encoding="ISO-8859-1") as f:
            pos = f.readlines()
            for i in pos:
                tokens = self.clean_str(i)
                all_tokens += tokens
        
        with open(osp.join(self.args.path, 'rt-polarity.neg.txt'), 'r', encoding="ISO-8859-1") as f:
            neg = f.readlines()
            for i in neg:
                tokens = self.clean_str(i)
                all_tokens += tokens
        
        self.vocab.build_vocab(all_tokens)
        
    def collate_fn(self, data):
        """
        add padding for text of various lengths
        Args:
            [(text(tensor), label(tensor)), ...]
        Returns:
            tensor, tensor : text, label
        """
        text, label = zip(*data)
        text = pad_sequence(text, batch_first=True, padding_value=self.vocab.word2idx['<pad>'])
        label = torch.stack(label, 0)
        return text, label
        
    def build_embeddings(self, word2vec):
        """
        Adapt pretrained vector to embedding vector
        """
        new_embeddings = []
        var_arr = word2vec.vectors.var(axis=0) # For OOV initialization, 300-dim vectors containing variances of each dimension
        for (word, idx) in self.vocab.word2idx.items():
            try:
                new_embeddings.append(word2vec[word])
            except KeyError:
                new_embeddings.append(np.random.uniform(low= var_arr, high= -var_arr))
        return torch.tensor(new_embeddings)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return torch.tensor(self.dataset[idx][0]), torch.tensor(float(self.dataset[idx][1]))
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset Builder')
    parser.add_argument('-b', '--batch_size', type=int, default=2)
    parser.add_argument('-p', '--path', type=str, default=r'C:\Users\Simon\ongoing_projects\torch_study\1_CNN\1_reference_code')
    args = parser.parse_args()
    
    dataset = CustomDataset(args)
    data_loader = DataLoader(dataset= dataset,
                             batch_size= args.batch_size,
                             collate_fn= dataset.collate_fn)
    
    for i in data_loader:
        print(i)
        break
    