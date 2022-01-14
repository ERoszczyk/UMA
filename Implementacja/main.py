from statistics import mode

import pandas as pd
import numpy as np
from Implementacja.Entities.Attribute import Attribute
from Implementacja.Entities.Node import Node


# S - Sample
# A - in S
def generate_tree(S):
    if all_samples_of_same_class(S):
        class_name = S.iloc[0]["class"]
        return Node(parent_conn=None, class_label=class_name)
    if all_attributes_the_same_or_empty(S):
        return Node(parent_conn=None, class_label=find_most_common_class(S))


def all_samples_of_same_class(S):
    return all(S.iloc[0]["class"] == row["class"] for index, row in S.iterrows())


def all_attributes_the_same_or_empty(S):
    if S.shape[0] == 0:
        return True
    for column in S.columns[1:]:
        attributes_in_column = S[column]
        if not all(attributes_in_column[0] == x for x in attributes_in_column):
            return False
    # if all columns were checked then all attributes are equal
    return True


def find_most_common_class(S):
    # if two have same count pick first
    li = []
    for val in S.iloc[:, -1:].values:
        li.append(val[0])
    return max(set(li), key=li.count)


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


if __name__ == '__main__':
    read_data = get_data_from_files('../Data/Car/car.data.small', '../Data/Car/car.c45-names')
    generate_tree(read_data[1])
    print("============= FINISHED =============")
