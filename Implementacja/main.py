import pandas as pd
from Entities.Attribute import Attribute
import json


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
    train_dataset = df.sample(frac=(4 / 7))  # random state is a seed value
    test_dataset = df.drop(train_dataset.index)

    return [train_dataset, test_dataset]


if __name__ == '__main__':
    # read_data = get_data_from_files('../Data/Car/car.data.small', '../Data/Car/car.c45-names')
    # read_data = get_data_from_files('../Data/adult/adult.data', '../Data/adult/adult.names')
    data = get_training_and_eval_sets('../Data/Car/car.data', '../Data/Car/car.c45-names')
    tree = Tree(data[0])
    tree.root = tree.generate_tree(tree.S)
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
