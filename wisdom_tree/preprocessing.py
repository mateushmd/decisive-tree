import pandas as pd
import numpy as np
from typing import Tuple

def split_data(data: pd.DataFrame, class_column_index=None, test_size=0.2, random_state=None) -> Tuple:
    if class_column_index is None:
        class_column_index = len(data.columns) - 1
    elif class_column_index < 0 or class_column_index > len(data.columns):
        raise ValueError("Invalid class column index")

    if random_state is not None:
        np.random.seed(random_state)

    y = data.iloc[:, class_column_index]
    X = data.drop(data.columns[class_column_index], axis=1)

    shuffled = np.random.permutation(len(X))
    test_size = int(len(X) * test_size)

    test_indices = shuffled[:test_size]
    train_indices = shuffled[test_size:]

    X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
    X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]

    return X_train, X_test, y_train, y_test