# trainer.py
import sys
sys.path.append(r"C:\Users\Simon\ongoing_projects\torch_study\1_CNN\3_packaging")

import os
import os.path as osp
import re
import pickle
import argparse
import random
import numpy as np
import functools
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split


from model import TextCNN
from dataloader import CustomDataset


SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Trainer:
    def __init__(self, args):
        self.args = args
        
        # data
        dataset = CustomDataset(args)
        self.collate_fn = dataset.collate_fn  # For zero-padding
        
        # For K-fold
        train_size = int(len(dataset) / args.cv_num)    
        
        print("Dataset size:", len(dataset))
        print("train size:", train_size)
        
            # Randomly split a dataset into non-overlapping new datasets of given lengths
        self.dataset_list = random_split(dataset,          
                                         [train_size for i in range(args.cv_num -1)] +\
                                         [len(dataset) - (args.cv_num - 1)*train_size]) # 각 데이터 size가 들어있는 리스트
        
        # arguments, loss
        self.vocab_size = len(dataset.vocab)  ## ??
        self.pad_idx = dataset.vocab.word2idx['<pad>']  ## ??
        self.embeddings = dataset.pretrained_embeddings
        self.criterion = nn.BCEWithLogitsLoss().to(device)
        
        # make directory if not exist data path
        if not osp.isdir(args.ck_path):
            os.makedirs(args.ck_path, exist_ok= True) # If exist_ok is False, FileExistsError is raised.
        
    def train(self):
        best_valid_loss = 1e9
        all_valid_loss, all_valid_acc = 0, 0
        
        # CV loop
        for i in range(self.args.cv_num):
            model = TextCNN(self.vocab_size, self.pad_idx, self.args).to(device)
            
            # model variations (cf. "rand" is default value)
            if self.args.mode == "static":
                model.static_embedding.weight.data.copy_(self.embeddings)
                model.static_embedding.weight.requires_grad = False
            elif self.args.mode == "non-static":
                model.static_embedding.weight.data.copy_(self.embeddings)
            elif self.args.mode == "multichannel":
                model.static_embedding.weight.data.copy_(self.embeddings)
                model.static_embedding.weight.requires_grad = False
                model.nonstatic_embedding.weight.data.copy_(self.embeddings)
            
            optimizer = optim.Adadelta(model.parameters())
            model.train()
           
            # generate train dataset
            print(f'>>> {i+1}th dataset is testset')  ## ??
            dataset = self.dataset_list.copy()
            del dataset[i]  # remove testset
            dataset = functools.reduce(lambda x, y: x + y, dataset) # Concatenate datasets consecutively.
            
            data_loader = DataLoader(dataset= dataset,
                                     batch_size= self.args.batch_size,
                                     shuffle= True,
                                     collate_fn= self.collate_fn)
           
            for epoch in range(self.args.epochs):   # Epoch loop
                pbar = tqdm(data_loader)
                
                for text, label in pbar:
                    text = text.to(device)
                    label = label.to(device)
                    
                    optimizer.zero_grad()
                    
                    predictions = model(text).squeeze(1)
                    loss = self.criterion(predictions, label)
                    acc = self._binary_accuracy(predictions, label)
                    
                    loss.backward()
                    optimizer.step()
                    
                    # max_norm_scaling
                    eps = 1e-7
                    param = model.fc.weight
                    norm = torch.norm(param) # l2_norm
                    if norm > self.args.l2_constraint:
                        param.data *= self.args.l2_constraint / (eps + norm)
                                        
                    pbar.set_description(f"loss : {loss.item():.4f}, acc : {acc.item():.4f}")

            
            valid_loss, valid_acc = self.evaluate(model, i)
            all_valid_loss += valid_loss.item()
            all_valid_acc += valid_acc.item()
            print(f'valid loss : {valid_loss.item():.3f}, valid acc : {valid_acc.item():.3f}')
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), 
                            osp.join(self.args.ck_path, f'{self.args.name}_best.pt'))

            if not self.args.cv:
                return
        
        print()
        print(f'Final loss : {all_valid_loss / self.args.cv_num:.3f}')
        print(f'Final acc : {all_valid_acc / self.args.cv_num:.3f}')
        
    def evaluate(self, model, cnt):
        """
        get loss, accuracy about test dataset
        Args:
            model(CNN) : trained model
            cnt(int) : test dataset's number in dataset
        Returns:
            loss(float), acc(float)
        """
        loss, acc = 0, 0
        
        model.eval()
        
        data_loader = DataLoader(dataset= self.dataset_list[cnt],
                                 batch_size= self.args.batch_size,
                                 shuffle= True,
                                 collate_fn= self.collate_fn)
        
        with torch.no_grad():
            for text, label in data_loader:
                text = text.to(device)
                label = label.to(device)
                predictions = model(text).squeeze(1)
                
                loss += self.criterion(predictions, label)
                acc += self._binary_accuracy(predictions, label)
        
        loss /= len(data_loader)
        acc /= len(data_loader)
        
        return loss, acc
    
    def _binary_accuracy(self, preds, y):
        # round predictions to the closest integer
        rounded_preds = torch.round(torch.sigmoid(preds))
        correct = (rounded_preds == y).float() #convert into float for division 
        acc = correct.sum() / len(correct)
        return acc
    
    def _smaple_data(self, loader):       #????
        while True:
            for batch in loader:
                yield batch
           
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trainer')
    parser.add_argument('--name', type=str, default='base')
    parser.add_argument('--mode', type=str, choices=['rand','static','non-static','multichannel'], default='rand')
    parser.add_argument('--ck_path', type=str, default=r'C:\Users\Simon\ongoing_projects\torch_study\1_CNN\checkpoint\\')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--path', type=str, default=r'C:\Users\Simon\ongoing_projects\torch_study\1_CNN\1_reference_code')
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--n_filters', type=int, default=100)
    parser.add_argument('--filter_sizes', type=list, default=[3,4,5])
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--cv_num', type=int, default=10)
    parser.add_argument('--l2_constraint', type=int, default=3)
    parser.add_argument("--cv", type=bool)
    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()              