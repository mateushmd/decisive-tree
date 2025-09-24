from collections import Counter
from typing import Any, List, Optional, Callable
import pandas as pd
import numpy as np
from . import util
from .node_data import NodeData
from treelib import Tree

class ID3:
    def __init__(self):
        self.tree = Tree()

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        return

    def build_tree(X: pd.DataFrame, y: pd.DataFrame, cur_node: str):
        best = min({ c: gain(X[c], y) for c in X.columns })
        feat_vals = np.unique(X[best]).tolist()
        for val in feat_vals:
            indexes = util.indexes_of(X[best], val)
            new_y = [y[i] for i in indexes]
            unique_classes = np.unique(new_y).tolist()
            node_data = None
            if len(unique_classes) == 1:
                node_data = NodeData(True, len(new_y), util.count_items(new_y), prediction=unique_classes[0])
            else:
                node_data = NodeData(False, len(new_y), util.count_items(new_y), feature=val)

            self.tree.create_node(val, val, parent=cur_node, data=node_data)
            return
            new_X = X.iloc[indexes]
            

    def gain(self, attr: pd.Series, y: pd.DataFrame) -> float:
        y_vals = y.values
        a_vals = attr.values
        freqs = util.relative_freq(a_vals)
        s = 0
        for k in freqs.keys():
            filtered = [y_vals[i] for i in util.indexes_of(a_vals, k)]
            s += freqs[k] * util.entropy(*list(util.relative_freq(filtered).values()))
        return util.entropy(*list(util.relative_freq(y_vals).values())) - s
