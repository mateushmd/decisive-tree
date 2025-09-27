from treelib import Tree

class Cart():
    def __init__(self):
        self._tree = None
    
    def plot(self):
        self._tree.show(data_property="tostr")
    
