from collections import Counter
from typing import Any, List, Optional, Callable
import pandas as pd
import numpy as np
from . import util
from .node_data import NodeData
from treelib import Tree

class ID3:
    def __init__(self):
        self._tree = None
        self._classes = None

    def plot(self):
        self._tree.show(data_property="tostr")

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        self._tree = Tree()
        self._classes = y.unique()
        self._tree.create_node(tag="Root", identifier="root")
        self._build_tree(X, y, "root")
        return self

    def _build_tree(self, X: pd.DataFrame, y: pd.Series, cur_id: str, branch=None):
        samples_count = len(y)
        samples_per_class = y.value_counts().reindex(self._classes, fill_value=0).tolist()
        
        if len(y.unique()) == 1:
            prediction = y.iloc[0]
            self._tree.get_node(cur_id).data = NodeData(True, samples_count, *samples_per_class, branch=branch, prediction=prediction)
            return

        if len(X.columns) == 0:
            majority_class = y.mode()[0]
            self._tree.get_node(cur_id).data = NodeData(True, samples_count, *samples_per_class, branch=branch, prediction=majority_class)
            return

        gains = { c: self._gain(X[c], y) for c in X.columns }
        best_feature = max(gains, key=gains.get)
        
        self._tree.get_node(cur_id).data = NodeData(False, samples_count, *samples_per_class, branch=branch, feature=best_feature)

        for value in X[best_feature].unique():
            child_id = f"{cur_id}_{value}"
            self._tree.create_node(tag=str(value), identifier=child_id, parent=cur_id)

            subset_indices = X[X[best_feature] == value].index
            new_X = X.loc[subset_indices].drop(columns=[best_feature])
            new_y = y.loc[subset_indices]

            self._build_tree(new_X, new_y, child_id, branch=value)
            
    def _gain(self, attr: pd.Series, y: pd.DataFrame) -> float:
        y_vals = y.values
        a_vals = attr.values
        freqs = util.relative_freq(a_vals)
        s = 0
        for k in freqs.keys():
            filtered = [y_vals[i] for i in util.indexes_of(a_vals, k)]
            s += freqs[k] * util.entropy(*list(util.relative_freq(filtered).values()))
        return util.entropy(*list(util.relative_freq(y_vals).values())) - s
