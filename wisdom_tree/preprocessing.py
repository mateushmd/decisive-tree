import pandas as pd
import numpy as np
from typing import Tuple

def split_data(X: pd.DataFrame, y: pd.Series, test_size=0.2, random_state=None) -> Tuple:
    if random_state is not None:
        np.random.seed(random_state)

    shuffled = np.random.permutation(len(X))
    test_size = int(len(X) * test_size)

    test_indices = shuffled[:test_size]
    train_indices = shuffled[test_size:]

    X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
    X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]

    return X_train, X_test, y_train, y_test