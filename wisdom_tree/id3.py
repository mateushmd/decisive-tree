from collections import Counter
from typing import Any, List
import pandas as pd
import numpy as np
from . import util

class ID3:
    def __init__(self):
        self.X = None
        self.y = None
        self.e_y = 0
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        

    def build_tree(X: pd.DataFrame, y: pd.DataFrame):
        return

    def gain(self, attr: pd.Series, y: pd.DataFrame) -> float:
        y_v = y.values
        e_y = util.entropy(*list(util.rfreq(y_v).values()))
        a_v = attr.values
        freqs = util.rfreq(a_v)
        s = 0
        for k in freqs.keys():
            filtered = [y_v[i] for i in util.indexes_of(a_v, k)]
            s += freqs[k] * util.entropy(*list(util.rfreq(filtered).values()))
        return self.e_y - s
