from wisdom_tree import ID3, util
import pandas as pd

data = pd.read_csv('data/restaurante.csv', sep=';')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

print(util.indexes_of(y, 'Sim'))

freqs = util.relative_frequency(X['Cliente'].values)
print({item: util.indexes_of(X['Cliente'].values, item) for item in freqs.keys()})