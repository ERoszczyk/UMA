from Implementacja.Entities.Node import Node
import math

from pandas import unique
from statistics import median
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
def entropy(data, unit='shannon'):
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


class Tree:
    def __init__(self, S, continuous_attributes=[]):
        self.S = S
        self.root = None
        self.continuous_attributes = continuous_attributes

    def generate_tree(self, S, attribute_dict=None):
        N = Node(parent_conn=None)
        if all_samples_of_same_class(S):
            class_name = S.iloc[0]["class"]
            return Node(parent_conn=None, class_label=class_name)
        if all_attributes_the_same_or_empty(S):
            return Node(parent_conn=None, class_label=find_most_common_class(S))
        n = self.attribute_selection_method(S)
        if attribute_dict is not None:
            attribute_dict[S.columns[n]] += S.shape[0]

        if S.columns[n] not in self.continuous_attributes:
            for value in unique(S.iloc[:, n]):
                N.split_crit = S.columns[n]
                # subset with an equal a_n value to "value"
                column_name = S.columns[n]
                S_n = S[S[column_name] == value]
                if S_n.empty:
                    attach_a_leaf(S, N, value)
                else:
                    self.attach_a_child(S_n.drop(S_n.columns[n], axis=1), N, value, attribute_dict=attribute_dict)

        else:
            split_value = median(S.iloc[:, n])
            column_name = S.columns[n]
            N.split_crit = column_name

            # attach a child for bigger values
            S_n = S[S[column_name] > split_value]
            if S_n.empty:
                attach_a_leaf(S, N, f">{split_value}")
            else:
                self.attach_a_child(S_n.drop(S_n.columns[n], axis=1), N, f">{split_value}", attribute_dict=attribute_dict)

            # attach a child for smaller values
            S_n = S[S[column_name] <= split_value]
            if S_n.empty:
                attach_a_leaf(S, N, split_value)
            else:
                self.attach_a_child(S_n.drop(S_n.columns[n], axis=1), N, f"<={split_value}", attribute_dict=attribute_dict)

        return N

    def attach_a_child(self, S_n, N, attribute_value, attribute_dict=None):
        child = self.generate_tree(S_n, attribute_dict=attribute_dict)
        N.attach_a_child(attribute_value, child)

    def attribute_selection_method(self, S):
        entropy_list = []

        for column in S:
            if column != 'class':
                if column not in self.continuous_attributes:
                    operation_array = S[column].values.tolist()
                else:
                    split_value = median(S[column].values.tolist())
                    simple_attrs = []
                    for val in S[column].values.tolist():
                        if val > split_value:
                            simple_attrs.append("+")
                        else:
                            simple_attrs.append("-")
                    operation_array = simple_attrs

                val_dict = {}
                for (val, class_) in zip(operation_array, S['class']):
                    if val not in val_dict:
                        val_dict[val] = []
                    val_dict[val].append(class_)

                ent_t = 0
                for attr in val_dict:
                    ent_i = 0
                    total = len(val_dict[attr])
                    counter = Counter(val_dict[attr])
                    for key in counter:
                        p = (counter[key] / total)
                        ent_i -= p * math.log(p, 2.) * total
                    ent_t += ent_i
                ent_t /= len(operation_array)
                entropy_list.append(ent_t)

        # for column in S:
        #     if column != 'class':
        #         entropy_list.append(entropy(S[column].values))
        return entropy_list.index(min(entropy_list))

    def __dict__(self):
        return self.root.to_dict()

    def to_dict(self):
        return self.__dict__()

    def predict(self, sample):
        return self.root.predict(sample, self.continuous_attributes)
