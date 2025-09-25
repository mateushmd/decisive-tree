from wisdom_tree import ID3
import pandas as pd
import numpy as np

data = pd.read_csv('data/restaurante.csv', sep=';')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

arvere = ID3().fit(X, y)
arvere.plot()