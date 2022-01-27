from statistics import mean

import numpy as np
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
    continuous_values = []
    for attr in attributes:
        if attr.values[0] == 'continuous':
            continuous_values.append(attr.name)
    return [classes, attributes, attribute_names, continuous_values]


def get_data_from_files(data_file, names_file):
    names_info = read_classes_and_attributes(names_file)
    data = pd.read_csv(data_file, sep=',', names=names_info[2])
    return [names_info, data]


def get_training_and_eval_sets(data_file, names_file, frac_arg=(4/7)):
    data = get_data_from_files(data_file, names_file)
    continuous_attributes = data[0][3]
    df = data[1]
    df = prepare_data(df, continuous_attributes)
    check_for_incomplete_data(df)
    train_dataset = df.sample(frac=frac_arg)  # random state is a seed value
    test_dataset = df.drop(train_dataset.index)

    return [train_dataset, test_dataset, continuous_attributes]


def prepare_data(df, continuous_attributes):
    dataframe = df.copy()
    for column in df.columns[1:]:
        attributes_in_column = df[column].values.tolist()

        if None in attributes_in_column:
            dataframe = dataframe.replace({column: None}, find_most_common_value_in_column(df, column, continuous_attributes))

        if " ?" in attributes_in_column:
            name = find_most_common_value_in_column(df, column, continuous_attributes)
            dataframe = dataframe.replace({column: " ?"}, name)

        if " " in attributes_in_column:
            dataframe = dataframe.replace({column: " "}, find_most_common_value_in_column(df, column, continuous_attributes))

    return dataframe


def check_for_incomplete_data(df):
    positive = False
    for column in df.columns[1:]:
        attributes_in_column = df[column].values.tolist()
        if None in attributes_in_column or " ?" in attributes_in_column or " " in attributes_in_column:
            positive = True
            print(f"INCOMPLETE DATA IN COLUMN: {column}")

    # if not positive:
    #     print("All data in df is complete!")


def find_most_common_value_in_column(df, column_name, continuous_attributes):
    # if two have same count pick first
    li = df[column_name].values.tolist()
    if column_name not in continuous_attributes:
        return max(set(li), key=li.count)
    else:
        return mean(list(map(int, li)))


def attribute_ranking(data_file, names_file, no_trees=100):
    data = get_training_and_eval_sets(data_file, names_file, frac_arg=1)
    # data = get_training_and_eval_sets('../Data/adult/adult.data', '../Data/adult/adult.names', frac_arg=1)
    continuous_attributes = data[2]
    data = data[0]
    attributes_ranking = {}
    for column in data.columns:
        if column != 'class':
            attributes_ranking[column] = 0

    for i in range(no_trees):
        split_dataset = data.sample(frac=(1 / 10), random_state=np.random.RandomState())
        tree = Tree(split_dataset, continuous_attributes)
        tree.root = tree.generate_tree(tree.S, attributes_ranking)

    attributes_ranking = sorted(attributes_ranking.items(), key=lambda x: x[1], reverse=True)
    print(attributes_ranking)


def main_task_for_tree():
    data = get_training_and_eval_sets('../Data/Car/car.data', '../Data/Car/car.c45-names', frac_arg=0.8)
    #data = get_training_and_eval_sets('../Data/adult/adult.data', '../Data/adult/adult.names', frac_arg=0.8)
    continuous_attributes = data[2]
    tree = Tree(data[0], continuous_attributes)
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
    print("=============================== ATTRIBUTE RANKING FOR CAR DATABASE ===============================")
    for i in range(10):
        attribute_ranking('../Data/Car/car.data', '../Data/Car/car.c45-names', no_trees=1000)

    print("=============================== ATTRIBUTE RANKING FOR ADULT DATABASE ===============================")
    for i in range(10):
        attribute_ranking('../Data/adult/adult.data', '../Data/adult/adult.names', no_trees=1000)
