from wisdom_tree import ID3, util
import pandas as pd
import numpy as np

data = pd.read_csv('data/restaurante.csv', sep=';')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

print(util.count_items(y))

tree = ID3()
tree.fit(X, y)

gains = {c: tree.gain(X[c], y) for c in X.columns}
k = min(gains)
print(np.unique(X[k], return_counts=True))

test = [0, 1, 0, 0, 1]
print(np.unique(test).tolist())
print(util.count_items(test))
print(np.unique(test, return_counts=True).counts)