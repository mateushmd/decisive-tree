from wisdom_tree import C45
import pandas as pd

data = pd.read_csv('data/weather-numeric.csv', sep=',')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

arvere = C45().fit(X, y)
arvere.plot()