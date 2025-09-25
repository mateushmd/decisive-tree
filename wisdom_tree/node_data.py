class NodeData:
    def __init__(self, is_leaf: bool, samples: int, *samples_values: int, branch=None, prediction=None, feature=None):
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.feature = feature
        self.tostr = ""
        if branch is not None:
            self.tostr += f"{branch}: "
        if self.is_leaf:
            self.tostr += f"Predict({self.prediction})"
        else:
            self.tostr += f"Split({self.feature})"
        self.tostr += f", samples={samples}, samples_values={samples_values}"