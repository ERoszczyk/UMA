import pandas as pd

# S - Sample
# A - L
from Implementacja.Entities.Attribute import Attribute


def generate_tree(S, A):
    pass


def all_samples_of_same_class(S):
    pass


def read_classes_and_attributes(filepath):
    f = open(filepath, 'r')
    lines = f.readlines()
    classes = None
    attributes = []
    for line in lines:
        # comments
        if line[0] == "|" or len(line) == 1:
            continue
        # names of classes
        elif classes is None:
            classes = line.replace("\n", "").split(', ')
        # attributes and its' values
        else:
            attribute = Attribute(line[:line.find(':')])
            for value in line[line.find(':')+1:].replace(" ", "").replace(".\n", "").split(','):
                attribute.add_value(value)
            attributes.append(attribute)
    f.close()
    return [classes, attributes]


def get_data_from_files(data_file, names_file):
    names_info = read_classes_and_attributes(names_file)
    data = pd.read_csv(data_file, sep=',', names=names_info[0])
    return [names_info, data]


if __name__ == '__main__':
    read_data = get_data_from_files('../Data/Car/car.data.csv', '../Data/Car/car.c45-names.txt')
    print("============= FINISHED =============")

