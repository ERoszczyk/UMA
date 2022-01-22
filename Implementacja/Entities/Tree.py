from Implementacja.Entities.Node import Node
import math

from pandas import unique
from collections import Counter


def all_samples_of_same_class(S):
    return all(S.iloc[0]["class"] == row["class"] for index, row in S.iterrows())


def all_attributes_the_same_or_empty(S):
    if S.shape[0] == 0:
        return True
    for column in S.columns[1:]:
        attributes_in_column = S[column]
        attr = attributes_in_column.iloc[0]
        if not all(attr == x for x in attributes_in_column):
            return False
    # if all columns were checked then all attributes are equal
    return True


def find_most_common_class(S):
    # if two have same count pick first
    li = []
    for val in S.iloc[:, -1:].values:
        li.append(val[0])
    return max(set(li), key=li.count)


def attach_a_leaf(S, N, attribute_value):
    child = Node(parent_conn=None, class_label=find_most_common_class(S))
    N.attach_a_child(attribute_value, child)


# copied from https://stackoverflow.com/questions/15450192/fastest-way-to-compute-entropy-in-python
def entropy(data, unit='natural'):
    base = {
        'shannon': 2.,
        'natural': math.exp(1),
        'hartley': 10.
    }

    if len(data) <= 1:
        return 0

    counts = Counter()

    for d in data:
        counts[d] += 1

    ent = 0

    probs = [float(c) / len(data) for c in counts.values()]
    for p in probs:
        if p > 0.:
            ent -= p * math.log(p, base[unit])

    return ent


def attribute_selection_method(S):
    entropy_list = []
    for column in S:
        if column != 'class':
            entropy_list.append(entropy(S[column].values))
    return entropy_list.index(min(entropy_list))


class Tree:
    def __init__(self, S):
        self.S = S
        self.root = None

    def generate_tree(self, S, attribute_dict=None):
        N = Node(parent_conn=None)
        if all_samples_of_same_class(S):
            class_name = S.iloc[0]["class"]
            return Node(parent_conn=None, class_label=class_name)
        if all_attributes_the_same_or_empty(S):
            return Node(parent_conn=None, class_label=find_most_common_class(S))
        n = attribute_selection_method(S)
        if attribute_dict is not None:
            attribute_dict[S.columns[n]] += S.shape[0]
        for value in unique(S.iloc[:, n]):
            N.split_crit = S.columns[n]
            # subset with an equal a_n value to "value"
            column_name = S.columns[n]
            S_n = S[S[column_name] == value]
            if S_n.empty:
                attach_a_leaf(S, N, value)
            else:
                self.attach_a_child(S_n.drop(S_n.columns[n], axis=1), N, value, attribute_dict=attribute_dict)
        return N

    def attach_a_child(self, S_n, N, attribute_value, attribute_dict=None):
        child = self.generate_tree(S_n, attribute_dict=attribute_dict)
        N.attach_a_child(attribute_value, child)

    def __dict__(self):
        return self.root.to_dict()

    def to_dict(self):
        return self.__dict__()

    def predict(self, sample):
        return self.root.predict(sample)
