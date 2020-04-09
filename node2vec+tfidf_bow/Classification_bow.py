import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from skorch import NeuralNetClassifier
import numpy as np
import pickle
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

torch.manual_seed(1234)

class Node2vecTfidfNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(Node2vecTfidfNN, self).__init__()
        self.hidden_size1 = int(input_size/2)
        self.output_size = output_size

        self.layer1 = nn.Linear(input_size, self.hidden_size1)
        self.layer2 = nn.Linear(self.hidden_size1, output_size)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.layer1(x))
        return self.layer2(x)

X = pickle.load(open("../data/Bow.pkl", "rb"))
y = pickle.load(open("../data/y.pkl", "rb"))
y_onehot = pickle.load(open("../data/y_onehot.pkl", "rb"))


print("Class percentage:")
for i in range(y.max()+1):
    print("Class {}: {}/{}={}%".format(i, len(y[y==i]), len(y), len(y[y==i])/len(y)*100))


params = {
    'lr': [0.1,0.05,0.01, 0.005,0.001],
    'max_epochs': [10, 15, 20, 25, 30,35,40,50,60],
    'optimizer': [optim.Adam, optim.AdamW, optim.RMSprop, optim.SGD]
}

K = 10



net = NeuralNetClassifier(Node2vecTfidfNN,
                          module__input_size=X.shape[1],
                          module__output_size=y.max()+1,
                          criterion=nn.CrossEntropyLoss,
                          iterator_train__shuffle=True,
                         )

print("Parameter search:")
skfold = StratifiedKFold(n_splits=K, shuffle=True, random_state=1234)
gs = GridSearchCV(net, params, cv=skfold, scoring=['f1_micro', 'f1_macro', 'accuracy'], refit='f1_macro')
gs.fit(X, y)


print("Best parameters:")
gs.best_params_

net = NeuralNetClassifier(Node2vecTfidfNN,
                          module__input_size=X.shape[1],
                          module__output_size=y.max()+1,
                          lr=gs.best_params_['lr'],
                          max_epochs=gs.best_params_['max_epochs'],
                          criterion=nn.CrossEntropyLoss,
                          iterator_train__shuffle=True,
                          optimizer = gs.best_params_['optimizer'],
                         )

print("Accuracy ({} cross validation)".format(K))
cross_val_score(net, X, y, scoring='accuracy', cv=skfold)




# ## OneVsRest 
print("OneVsRest with best parameters")

net = NeuralNetClassifier(Node2vecTfidfNN,
                          module__input_size=X.shape[1],
                          module__output_size=2,
                          lr=gs.best_params_['lr'],
                          max_epochs=gs.best_params_['max_epochs'],
                          criterion=nn.CrossEntropyLoss,
                          iterator_train__shuffle=True,
                          optimizer = gs.best_params_['optimizer'],
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
        X_train, X_val, y_train, y_val = X[idx_train], X[idx_val], y_prime[idx_train], y_prime[idx_val]
        print("*** class {} fold {}".format(i, k))
        ratio_matrix[i][k][0] = sum(y_train)/len(y_train)
        ratio_matrix[i][k][1] = sum(y_val)/len(y_val)
        net.initialize()
        y_pred = net.fit(X_train,y_train).predict(X_val)
        f1_matrix[i][k] = f1_score(y_val, y_pred)
        precision_matrix[i][k] = precision_score(y_val, y_pred)
        recall_matrix[i][k] = precision_score(y_val, y_pred)
        accuracy_matrix[i][k] = accuracy_score(y_val, y_pred)


for i in range(y.max()+1):
    for k in range(K):
        print("Class {} Fold {}: train ratio:{:.2f}% val ratio:{:.2f}%.".format(i,k+1,ratio_matrix[i][k][0]*100,ratio_matrix[i][k][1]*100))

print("Result matrix of each split and each class\n the i-th row and j-th column represents the corresponding result of class number i and split number j")
print("F1 score")
print(f1_matrix)
print("Precision score")
print(precision_matrix)
print("Recall score")
print(recall_matrix)
print("Accuracy score")
print(accuracy_matrix)
