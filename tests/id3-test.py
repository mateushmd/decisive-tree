from wisdom_tree import ID3, util
import pandas as pd

data = pd.read_csv('data/restaurante.csv', sep=';')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

tree = ID3()
tree.fit(X, y)

gains = {c: tree.gain(X[c], y) for c in X.columns}
print(gains)
print(min(gains, key=gains.get))