import math
import time
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os

import pickle
import scipy.sparse as sp
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from skorch import NeuralNetClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


X = pickle.load(open("X_tfidf_matrix.pkl", "rb"))
y_onehot = pickle.load(open("y_onehot.pkl", "rb"))
y = np.argmax(y_onehot, axis=1)

X[:,:100]=pickle.load(open("../data/Bow.pkl", "rb"))


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


adj = pickle.load(open("Adj_sparse.pkl", "rb"))
# build symmetric adjacency matrix
adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
# mytrick: add epsilon, 
#epsilon = 1
adj = normalize(adj + sp.eye(adj.shape[0]))

# 2 order develop of g * x
#adj = normalize(adj c+ 0.001*sp.eye(adj.shape[0]))
#adj = adj + 2*adj.dot(adj)
adj = sparse_mx_to_torch_sparse_tensor(adj)


X = normalize(X)


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):

        support = torch.mm(input, self.weight)

        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' ('                + str(self.in_features) + ' -> '                + str(self.out_features) + ')'



class GCN(nn.Module):
    def __init__(self, nfeat, nclass, adj, allfeatures, dropout = 0.0):
        super(GCN, self).__init__()
        self.adj = adj
        self.allfeatures = allfeatures
        self.gc1 = GraphConvolution(nfeat, int(nfeat/2))
        
        self.gc2 = GraphConvolution(int(nfeat/2), nclass)
        self.dropout = dropout

    def forward(self, idx):
        x = F.relu(self.gc1(self.allfeatures, self.adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, self.adj)
        
        return x[idx.long()]


net = NeuralNetClassifier(GCN,
                          module__nfeat=X.shape[1],
                          module__nclass=y.max()+1,
                          module__adj=adj,
                          module__allfeatures=torch.FloatTensor(X),
                          criterion=nn.CrossEntropyLoss,
                          iterator_train__shuffle=True,
                          batch_size=-1
                         )



params = {
    'lr': [0.1,0.05,0.01, 0.005,0.001],
    'max_epochs': [ 20, 25, 30,40,50,60],
    'optimizer': [optim.Adam, optim.AdamW, optim.RMSprop]
}

K = 10
skfold = StratifiedKFold(n_splits=K, shuffle=True, random_state=1234)


print("Parameter search:")
gs = GridSearchCV(net, params, cv=skfold, scoring=['f1_micro', 'f1_macro', "accuracy"], refit="f1_macro")
gs.fit(np.arange(0,len(X)), y)

print("Best parameters:")
gs.best_params_

net = NeuralNetClassifier(GCN,
                          module__nfeat=X.shape[1],
                          module__nclass=y.max()+1,
                          module__adj=adj,
                          module__allfeatures=torch.FloatTensor(X),
                          lr=gs.best_params_['lr'],
                          max_epochs=gs.best_params_['max_epochs']+20,
                          optimizer = gs.best_params_['optimizer'],
                          criterion=nn.CrossEntropyLoss,
                          iterator_train__shuffle=True,
                          batch_size=-1
                         )


f1macro = np.mean(cross_val_score(net, np.arange(len(X)), y, scoring='f1_macro', cv=skfold))
net.initialize()
f1micro = np.mean(cross_val_score(net, np.arange(len(X)), y, scoring='f1_micro', cv=skfold))
net.initialize()
accuracy = np.mean(cross_val_score(net, np.arange(len(X)), y, scoring='accuracy', cv=skfold))

print("Results with best parameters")
print("F1 macro:{}\nF1_micro:{}\nAccuracy:{}".format(f1macro, f1micro, accuracy))



print("OneVSRest Results with best parameters")
net = NeuralNetClassifier(GCN,
                          module__nfeat=X.shape[1],
                          module__nclass=2,
                          module__adj=adj,
                          module__allfeatures=torch.FloatTensor(X),
                          lr=gs.best_params_['lr'],
                          max_epochs=gs.best_params_['max_epochs'],
                          optimizer = gs.best_params_['optimizer'],
                          criterion=nn.CrossEntropyLoss,
                          iterator_train__shuffle=True,
                          batch_size=-1
                         )


ratio_matrix = np.zeros((y_onehot.shape[1], K, 2))
f1_matrix = np.zeros((y_onehot.shape[1], K))
precision_matrix = np.zeros((y_onehot.shape[1], K))
recall_matrix = np.zeros((y_onehot.shape[1], K))
accuracy_matrix = np.zeros((y_onehot.shape[1], K))

for k, idx in enumerate(skfold.split(X, y.reshape(-1))):
    idx_val, idx_train = idx
    for i in range(y.max()+1):
        y_prime = np.zeros(y.shape[0])
        y_prime[y==i] = 1
        y_prime = y_prime.astype(np.longlong)
        y_train, y_val = y_prime[idx_train], y_prime[idx_val]
        print("*** class {} fold {}".format(i, k))
        ratio_matrix[i][k][0] = sum(y_train)/len(y_train)
        ratio_matrix[i][k][1] = sum(y_val)/len(y_val)
        net.initialize()
        y_pred = net.fit(idx_train,y_train).predict(idx_val)
        f1_matrix[i][k] = f1_score(y_val, y_pred)
        precision_matrix[i][k] = precision_score(y_val, y_pred)
        recall_matrix[i][k] = precision_score(y_val, y_pred)
        accuracy_matrix[i][k] = accuracy_score(y_val, y_pred)



for i in range(y.max()+1):
    for k in range(K):
        print("Class {} Fold {}: train ratio:{:.2f}% val ratio:{:.2f}%.".format(i,k+1,ratio_matrix[i][k][0]*100,ratio_matrix[i][k][1]*100))

print("Result matrix:\n i-th row and j-th column represents the corresponding result of i-th class and j-th split.")
print("F1 score:")
print(f1_matrix)
print("Precision score:")
print(precision_matrix)
print("Recall score")
print(recall_matrix)
print("Accuracy score:")
print(accuracy_matrix)

