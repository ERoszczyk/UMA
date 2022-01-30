from Implementacja.Entities.Tree import Tree


class Forest:
    def __init__(self, num_trees=100):
        self.roots = []
        self.num_trees = num_trees
        self.data = None

    def generate_forest(self, data):
        self.data = data
        for i in range(self.num_trees):
            # make smaller, more specialized trees
            train_dataset = data.sample(frac=(1 / 10))
            tree = Tree(train_dataset)
            tree.root = tree.generate_tree(tree.S)
            self.roots.append(tree.root)

    def predict(self, sample):
        predictions = []
        for tree in self.roots:
            predictions.append(tree.predict(sample))
        return max(set(predictions), key=predictions.count)

    def list_of_possible_classes(self):
        return self.data['class'].unique().tolist()
