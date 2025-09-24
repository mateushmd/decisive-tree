class NodeData:
    def __init__(self, is_leaf: bool, samples: int, *samples_values: int, prediction=None, feature=None):
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.feature = feature

    def __repr__(self):
        s = ""
        if self.is_leaf:
            s = f"Predict: {self.prediction}"
        else:
            s = f"Split on: {self.feature}"
        s += f", samples={samples}, samples_values={samples_values}"