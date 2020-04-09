import pickle
import networkx as nx
import numpy as np
import random
from collections import Counter
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold


np.random.seed(1234)

graph = pickle.load(open("../data/graph_withnodelabel.pkl", "rb"))

label_dict_onehot = pickle.load(open("../data/label_compatible.pkl", "rb"))
label_dict = {}
for k,v in label_dict_onehot.items():
    label_dict[k] = np.argmax(v)



num_class = np.max(list(label_dict.values())) + 1
def neighbor_label(x):
    return label_dict[x]


def LP_withlabel(nodes_train, nodes_val, label_dict, graph):
    # Create the dictionary labels
    label = {}
    for n in nodes_val:
        label[n] = -1
    ##########################################
    # Label propagation?
    cont = True
    while cont:
        cont = False
        nodes = list(graph)
        random.shuffle(nodes)

        # Calculate the label for each node
        for node in nodes:
            if node in nodes_train or label[node] != -1: #  in training dataset or has been labeled
                continue
            
            # get neighbor, not including the node itself
            neighbor_list = np.array([])
            for v in graph[node]:
                if v != node:
                    neighbor_list = np.append(neighbor_list, v)
            
            # no other node as neighbor, select randomly one label
            if len(neighbor_list) == 0:
                label[node] = np.random.randint(0, num_class)
                continue
            
            # get neighborhood's label
            neighbor_label_list = np.array(list(map(neighbor_label, neighbor_list)))
            
            
            # if all the neighbor are not labeled
            if np.all(neighbor_label_list == -1):
                continue
            else:
                # consider only neighbor with label
                neighbor_list = neighbor_list[neighbor_label_list!=-1]
            

            # Get label frequencies. Depending on the order they are processed
            # in some nodes with be in t and others in t-1, making the
            # algorithm asynchronous.
            label_freq = Counter()
            for v in neighbor_list:
                
                label_freq.update({label_dict[v]: graph.edges[node, v]['weight']})
            # Choose the label with the highest frecuency. If more than 1 label
            # has the highest frecuency choose one randomly.
            max_freq = max(label_freq.values())
            best_labels = [label for label, freq in label_freq.items()
                            if freq == max_freq]

            label[node] = random.choice(best_labels)
            cont = True
    #############################################
    return label


K = 10
skfold = StratifiedKFold(n_splits=K, shuffle=True, random_state=1234)
node_list = np.array(graph.nodes())
y = np.array(list(map(neighbor_label, node_list)))


scores = np.zeros((K, 3))
for k, idx in enumerate(skfold.split(np.arange(len(graph)), y)):
    idx_train, idx_val = idx
    res = LP_withlabel(node_list[idx_train], node_list[idx_val], label_dict, graph)
    y_pred = []
    y_val = []
    for n in node_list[idx_val]:
        y_val.append(label_dict[n])
        y_pred.append(res[n])
    scores[k][0]=accuracy_score(y_val, y_pred)
    scores[k][1]=f1_score(y_val, y_pred, average='micro')
    scores[k][2]=f1_score(y_val, y_pred, average='macro')

print("Result of {} cross validation:".format(K))
print("f1 macro score: ",np.mean(scores[:,2]))
print("f1 micro score: ",np.mean(scores[:,1]))
print("accuracy score: ",np.mean(scores[:,0]))

