import pandas as pd
from Entities.Attribute import Attribute
import json

from Implementacja.Entities.Forest import Forest
from Implementacja.Entities.Tree import Tree


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
    train_dataset = df.sample(frac=(8 / 10))  # random state is a seed value
    test_dataset = df.drop(train_dataset.index)

    return [train_dataset, test_dataset]


def attribute_ranking(no_trees=100):
    data = get_data_from_files('../Data/Car/car.data', '../Data/Car/car.c45-names')[1]
    attributes_ranking = {}
    for column in data.columns:
        if column != 'class':
            attributes_ranking[column] = 0

    for i in range(no_trees):
        split_dataset = data.sample(frac=(4 / 7))
        tree = Tree(split_dataset)
        tree.root = tree.generate_tree(tree.S, attributes_ranking)

    attributes_ranking = sorted(attributes_ranking.items(), key=lambda x: x[1], reverse=True)
    print(attributes_ranking)


def main_task_for_tree():
    data = get_training_and_eval_sets('../Data/Car/car.data', '../Data/Car/car.c45-names')
    tree = Tree(data[0])
    tree.root = tree.generate_tree(tree.S)
    tree_in_dict = tree.to_dict()
    tree_in_json = json.dumps(tree_in_dict)
    print(tree_in_json)
    print("Copy above line(json) to https://vanya.jp.net/vtree/")
    get_prediction_results(tree, data[1])


def main_task_for_forest():
    data = get_training_and_eval_sets('../Data/Car/car.data', '../Data/Car/car.c45-names')
    forest = Forest(100)
    forest.generate_forest(data[0])
    get_prediction_results(forest, data[1])


def get_prediction_results(predicting_obj, test_data):
    correct = 0
    incorrect = 0
    error = 0
    error_samples = []
    for index, row in test_data.iterrows():
        try:
            prediction = predicting_obj.predict(row)
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


if __name__ == '__main__':
    for i in range(10):
        attribute_ranking(no_trees=1000)
