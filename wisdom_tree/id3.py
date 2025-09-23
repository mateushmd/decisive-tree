from collections import Counter
from typing import Any, List
import pandas as pd
import numpy as np
from . import util

class ID3:
    def __init__(self):
        self.y = None
        self.e_y = 0
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        self.y = y.values
        self.e_y = util.entropy(util.relative_frequency(y).values)
    
    def gain(self, attr: List[Any]) -> float:
        freqs = util.relative_frequency(attr)
        y_freqs = {item: util.indexes_of(attr, item) for item in freqs.keys()}
        return self.e_y - sum(())
