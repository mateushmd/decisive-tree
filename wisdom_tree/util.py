from collections import Counter
from typing import List, Any
import numpy as np

def relative_frequency(x: List[Any]) -> List[float]:
    return {item: count / len(x) for item, count in Counter(x).items()}

def entropy(*p: float) -> float:
    return sum((-pi * np.log2(pi)) for pi in p)

def indexes_of(arr: List[Any], x: Any) -> List[int]:
    return list(filter(lambda i: arr[i] == x, range(len(arr))))