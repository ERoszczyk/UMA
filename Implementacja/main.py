from statistics import mean

import numpy as np
import pandas as pd
from Entities.Attribute import Attribute
import json

from Implementacja.Entities.Forest import Forest
from Implementacja.Entities.Tree import Tree
from timeit import default_timer as timer
import matplotlib.pylab as plt


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


def get_training_and_eval_sets(data_file, names_file, frac_arg=(4 / 7)):
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
            dataframe = dataframe.replace({column: None},
                                          find_most_common_value_in_column(df, column, continuous_attributes))

        if " ?" in attributes_in_column:
            name = find_most_common_value_in_column(df, column, continuous_attributes)
            dataframe = dataframe.replace({column: " ?"}, name)

        if " " in attributes_in_column:
            dataframe = dataframe.replace({column: " "},
                                          find_most_common_value_in_column(df, column, continuous_attributes))

    return dataframe


def check_for_incomplete_data(df):
    for column in df.columns[1:]:
        attributes_in_column = df[column].values.tolist()
        if None in attributes_in_column or " ?" in attributes_in_column or " " in attributes_in_column:
            print(f"INCOMPLETE DATA IN COLUMN: {column}")


def randomly_permute_column(df, column_name):
    df_copy = df.copy()
    df_copy[column_name] = np.random.permutation(df_copy[column_name])
    return df_copy


def find_most_common_value_in_column(df, column_name, continuous_attributes):
    # if two have same count pick first
    li = df[column_name].values.tolist()
    if column_name not in continuous_attributes:
        return max(set(li), key=li.count)
    else:
        return mean(list(map(int, li)))


def attribute_ranking_by_count(data_file, names_file, no_trees=100):
    data = get_training_and_eval_sets(data_file, names_file, frac_arg=1)
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


def attribute_ranking_by_attribute_poisoning(data_file, names_file, no_trees=100):
    data = get_training_and_eval_sets(data_file, names_file, frac_arg=(8 / 10))
    continuous_attributes = data[2]
    train_data = data[0]
    test_data = data[1]
    attributes_ranking = {}
    trained_models = []
    # train no_trees models
    for model_index in range(no_trees):
        split_dataset = train_data.sample(frac=(1 / 10), random_state=np.random.RandomState())
        tree = Tree(split_dataset, continuous_attributes)
        tree.root = tree.generate_tree(tree.S)
        trained_models.append(tree)

    # check on control sample
    control_sample_results = []
    for tree in trained_models:
        control_sample_results.append(get_prediction_results(tree, test_data, if_print=False))
    attributes_ranking['control'] = mean(control_sample_results)
    for column in train_data.columns:
        if column != 'class':
            poisoned_test_df = randomly_permute_column(test_data, column)
            attribute_poison_results = []
            for tree in trained_models:
                attribute_poison_results.append(get_prediction_results(tree, poisoned_test_df, if_print=False))
            attributes_ranking[column] = mean(attribute_poison_results)

    attributes_ranking = sorted(attributes_ranking.items(), key=lambda x: x[1], reverse=True)
    print(attributes_ranking)


def main_task_for_tree():
    data = get_training_and_eval_sets('../Data/Car/car.data', '../Data/Car/car.c45-names', frac_arg=(4 / 7))
    # data = get_training_and_eval_sets('../Data/adult/adult.data', '../Data/adult/adult.names', frac_arg=(8/10))
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


def get_prediction_results(predicting_obj, test_data, if_print=True):
    correct = 0
    incorrect = 0
    error = 0
    error_samples = []
    FP = {}
    FN = {}
    TP = {}
    TN = {}
    classes = predicting_obj.list_of_possible_classes()
    for index, row in test_data.iterrows():
        try:
            prediction = predicting_obj.predict(row)
        except Exception as e:
            error += 1
            error_samples.append(row)
            continue
        if prediction == row['class']:
            correct += 1
            if not row['class'] in TP:
                TP[row['class']] = 0
            TP[row['class']] += 1
            for class_ in classes:
                if class_ != prediction:
                    if not class_ in TN:
                        TN[class_] = 0
                    TN[class_] += 1
        else:
            incorrect += 1
            if not row['class'] in FN:
                FN[row['class']] = 0
            FN[row['class']] += 1

            if not prediction in FP:
                FP[prediction] = 0
            FP[prediction] += 1
    if if_print:
        print(f"Correct = {correct} | Incorrect = {incorrect} | Overall_accuracy = {correct / (correct + incorrect)}")
        for key in TP:
            print(f"    ========= {key} =========")
            TP_i = TP[key]
            FN_i = FN[key]
            FP_i = FP[key]
            TN_i = TN[key]
            print(f"        TP = {TP_i} | FN = {FN_i} | FP = {FP_i} | TN = {TN_i}")
            print(f"        TPR = {TP_i / (TP_i + FN_i)}")
            print(f"        TNR = {TN_i / (TN_i + FP_i)}")
            print(f"        PPV = {TP_i / (TP_i + FP_i)}")
            print(f"        ACC = {(TP_i + TN_i) / (TP_i + TN_i + FP_i + FN_i)}")
            print(f"        F1 = {2 * TP_i / (2 * TP_i + FP_i + FN_i)}")
        print(f" Error = {error}")
        print("============= FINISHED =============")
    return correct / (correct + incorrect)


def calculate_timing_performance(bins, data_file, names_file):
    data = get_training_and_eval_sets(data_file, names_file, frac_arg=1)
    continuous_attributes = data[2]
    df = data[0]
    test_data = df.head(100)
    elapsed_creation_time = {}
    elapsed_prediction_time = {}
    bin_size = df.shape[0] / bins
    for i in range(bins):
        frac = (i + 1) / bin_size
        if frac > 1:
            frac = 1
        train_data = df.sample(frac=frac, random_state=np.random.RandomState())
        start = timer()
        tree = Tree(train_data, continuous_attributes)
        tree.root = tree.generate_tree(tree.S)
        end = timer()
        elapsed_creation_time[train_data.shape[0]] = end - start
        elapsed_time = 0
        for index, row in test_data.iterrows():
            start = timer()
            tree.predict(row)
            end = timer()
            elapsed_time += end - start
        elapsed_prediction_time[train_data.shape[0]] = elapsed_time

    lists = sorted(elapsed_creation_time.items())  # sorted by key, return a list of tuples
    x, y = zip(*lists)  # unpack a list of pairs into two tuples

    plt.plot(x, y)
    plt.xlabel("Training set size")
    plt.ylabel("Tree generation time")
    plt.show()

    lists = sorted(elapsed_prediction_time.items())  # sorted by key, return a list of tuples

    x, y = zip(*lists)  # unpack a list of pairs into two tuples

    plt.plot(x, y)
    plt.xlabel("Training set size")
    plt.ylabel("Sample prediction time")
    plt.show()

    return [elapsed_creation_time, elapsed_prediction_time]


if __name__ == '__main__':
    main_task_for_tree()
# calculate_timing_performance(100)
# print("=============================== ATTRIBUTE RANKING FOR CAR DATABASE ===============================")
# for i in range(10):
#     attribute_ranking_by_count('../Data/Car/car.data', '../Data/Car/car.c45-names', no_trees=1000)
#
# print("=============================== ATTRIBUTE RANKING FOR ADULT DATABASE ===============================")
# for i in range(10):
#     attribute_ranking_by_count('../Data/adult/adult.data', '../Data/adult/adult.names', no_trees=1000)
# print("=============================== ATTRIBUTE RANKING BY POISONING FOR CAR DATABASE"
#       " ===============================")
# for i in range(10):
#     attribute_ranking_by_attribute_poisoning('../Data/Car/car.data', '../Data/Car/car.c45-names', no_trees=1000)
#
# print("=============================== ATTRIBUTE RANKING BY POISONING FOR ADULT DATABASE"
#       " ===============================")
# for i in range(10):
#     attribute_ranking_by_attribute_poisoning('../Data/adult/adult.data', '../Data/adult/adult.names', no_trees=1000)
