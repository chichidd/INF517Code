# INF517Code
This repository includes Python code for the project of graph-based webpage classification supervised by michalis vazirgiannis and Christos Xypolopoulos.
The data is labeled part of frech webpage domain graph crawled by Christos Xypolopoulos and contains 1,421 nodes and 4,172 edges. We train classifiers using different
methods in order to automatically catecorize the rest domain nodes of the original graph which has nearly 28,868 nodes and 133,526 edge into the 8 categories below:
* Politics/Government/Law
* Health/Medical
* Entertainment
* Business/Finance
* Tech/Science
* Education/Research
* Sports
* None ( for pages that do not belong to any of the topics listed above)

The final goal is to generate topical french language models (Word2vec, etc) using the crawled webpage text.\

## Run the code:
Run the command in the root directory: `python [script.py]`, where `[script.py]` is the script using the specific method. Note that we use `sklearn.model_selection.GridSearchCV` to auto-select best parameters of models.

## Requirements
```
torch==1.4.0
networkx==2.4.0
numpy==1.18.1
skorch==0.7.0
sklearn==0.22.1
```
