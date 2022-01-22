class Forest:
    def __init__(self, data, num_trees=100):
        self.roots = []
        self.data = data
        self.num_trees = num_trees

    def generate_forest(self):
        for i in range(self.num_trees):
            # make smaller, more specialized trees
            train_dataset = self.data.sample(frac=(1 / 7))
            self.roots.append()
