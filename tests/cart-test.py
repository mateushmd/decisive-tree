from wisdom_tree import Cart
import pandas as pd

data = pd.read_csv('data/weather-numeric.csv', sep=',')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

arvere = Cart().fit(X, y)
arvere.plot()