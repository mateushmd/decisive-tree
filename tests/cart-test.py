from wisdom_tree import CART
import pandas as pd

data = pd.read_csv('data/weather-numeric.csv', sep=',')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

arvere = CART().fit(X, y)
arvere.plot()