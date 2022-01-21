import math
import sys
import threading
from statistics import mode

import pandas as pd
import numpy as np
from pandas import unique

from Implementacja.Entities.Attribute import Attribute
from Implementacja.Entities.Node import Node
import scipy.stats
from collections import Counter
import json
from random import randrange


# S - Sample
# A - in S
def generate_tree(S):
    N = Node(parent_conn=None)
    if all_samples_of_same_class(S):
        class_name = S.iloc[0]["class"]
        return Node(parent_conn=None, class_label=class_name)
    if all_attributes_the_same_or_empty(S):
        return Node(parent_conn=None, class_label=find_most_common_class(S))
    n = attribute_selection_method(S)
    for value in unique(S.iloc[:, n]):
        N.split_crit = S.columns[n]
        # subset with an equal a_n value to "value"
        column_name = S.columns[n]
        S_n = S[S[column_name] == value]
        if S_n.empty:
            attach_a_leaf(S, N, value)
        else:
            attach_a_child(S_n.drop(S_n.columns[n], axis=1), N, value)
    return N


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


def attribute_selection_method(S):
    entropy_list = []
    for column in S:
        if column != 'class':
            entropy_list.append(entropy(S[column].values))
    return entropy_list.index(min(entropy_list))


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


def attach_a_leaf(S, N, attribute_value):
    child = Node(parent_conn=None, class_label=find_most_common_class(S))
    N.attach_a_child(attribute_value, child)


def attach_a_child(S_n, N, attribute_value):
    child = generate_tree(S_n)
    N.attach_a_child(attribute_value, child)


def read_classes_and_attributes(filepath):
    f = open(filepath, 'r')
    lines = f.readlines()
    classes = None
    attributes = []
    attribute_names = []
    for line in lines:
        # comments
        if line[0] == "|" or len(line) == 1:
            continue
        # names of classes
        elif classes is None:
            classes = line.replace("\n", "").split(', ')
        # attributes and its' values
        else:
            attribute_name = line[:line.find(':')]
            attribute = Attribute(attribute_name)
            attribute_names.append(attribute_name)
            for value in line[line.find(':') + 1:].replace(" ", "").replace(".\n", "").split(','):
                attribute.add_value(value)
            attributes.append(attribute)
    f.close()
    attribute_names.append("class")
    return [classes, attributes, attribute_names]


def get_data_from_files(data_file, names_file):
    names_info = read_classes_and_attributes(names_file)
    data = pd.read_csv(data_file, sep=',', names=names_info[2])
    return [names_info, data]


def get_training_and_eval_sets(data_file, names_file):
    data = get_data_from_files(data_file, names_file)
    df = data[1]
    train_dataset = df.sample(frac=(4 / 7))  # random state is a seed value
    test_dataset = df.drop(train_dataset.index)

    return [train_dataset, test_dataset]


if __name__ == '__main__':
    # read_data = get_data_from_files('../Data/Car/car.data.small', '../Data/Car/car.c45-names')
    # read_data = get_data_from_files('../Data/adult/adult.data', '../Data/adult/adult.names')
    data = get_training_and_eval_sets('../Data/Car/car.data', '../Data/Car/car.c45-names')
    tree = generate_tree(data[0])
    tree_in_dict = tree.to_dict()
    tree_in_json = json.dumps(tree_in_dict)
    print(tree_in_json)
    print("Copy above line(json) to https://vanya.jp.net/vtree/")

    correct = 0
    incorrect = 0
    error = 0
    error_samples = []
    for index, row in data[1].iterrows():
        try:
            prediction = tree.predict(row)
        except Exception as e:
            error += 1
            error_samples.append(row)
            continue
        if prediction == row['class']:
            correct += 1
        else:
            incorrect += 1
    print(f"Correct = {correct} | Incorrect = {incorrect} | Score = {correct / (correct + incorrect)}")
    print(f"Error = {error}")
    print("============= FINISHED =============")
