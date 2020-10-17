import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import sys
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import torch.optim as optim

class CustomDataset(Dataset):
    def __init__(self, x_local, x_global, y):
        self.x_local = x_local
        self.x_global = x_global
        self.y = y

    def __len__(self):
        return len(self.x_local)

    def __getitem__(self,index):
        X_local = self.x_local[index]
        X_global = self.x_global[index]
        Y = self.y[index]
        return X_local, X_global, Y

class Multichannel(nn.Module):
    def __init__(self, batch_size, local_dim, global_dim, label_dim, hidden_dim, activation):
        super(Multichannel, self).__init__()
        self.local_channel = nn.Linear(
            in_features = local_dim[1],
            out_features = hidden_dim
            )
        self.global_channel = nn.Linear(
            in_features = global_dim[1],
            out_features = hidden_dim
            )
        self.siamese_channel = nn.Linear(
            in_features = local_dim[1],
            out_features = hidden_dim
            ) # same dim
        self.out = nn.Linear(
            in_features = 3*hidden_dim, # fix later
            out_features = 1
        )
        self.sigmoid_layer = nn.Sigmoid()

        activation_dict ={
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU()
        }

        self.activation = activation_dict[activation]
        # self.siamese_channel = nn.Linear(local_dim[0], local_dim[1])

    def forward_one(self, x):
        x = self.siamese_channel(x)
        return x

    def forward(self, x_local, x_global):
        h1 = self.activation(self.local_channel(x_local))
        h2 = self.activation(self.global_channel(x_global))

        # shared weights
        h3a = self.activation(self.forward_one(x_local))
        h3b = self.activation(self.forward_one(x_global))

        d = torch.abs(h3a-h3b)

        c = torch.cat(tensors = (h1,h2,d), dim= 1)
        # print(c.shape)
        # sys.exit()
        e = self.out(c)
        # print(h1.shape)
        # print(h2.shape)
        # print(h3a.shape)
        # print(h3b.shape)
        # print(d.shape)
        # print(c.shape)
        # print(e.shape)
        # sys.exit()

        # softmax_layer = nn.Softmax(dim=1) # why use softmax?
        y_pred = self.sigmoid_layer(e)

        # print("Predictions:", y_pred_labels)

        return y_pred
