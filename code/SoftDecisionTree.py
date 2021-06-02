import numpy as np


def membership_output(middle_value, omega, input_value):
    if input_value < middle_value - omega / 2:
        return 0
    elif input_value > middle_value + omega / 2:
        return 1
    else:
        prob = (middle_value + omega / 2 - input_value) / omega
        random_value = np.random.uniform(0, 1)
        if random_value <= prob:
            return 0
        else:
            return 1


def classify(input_tree, test_data, n):
    first_string = next(iter(input_tree))
    second_dict = input_tree[first_string]
    feature_index = attribute.index(first_string)
    global class_label
    for key in second_dict.keys():
        if membership_output(eval(key[1:]), eval(key[1: ]) / n, test_data[feature_index]) == 0 and key[0] == '<':
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key], test_data, n)
            else:
                class_label = classification.index(second_dict[key])
        elif membership_output(eval(key[1: ]), eval(key[1: ]) / n, test_data[feature_index]) == 1 and key[0] == '>':
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key], test_data, n)
            else:
                class_label = classification.index(second_dict[key])
    return class_label


attribute = ['sepal length', 'sepal width', 'petal length', 'petal width']
classification = ['Setosa', 'Versicolour', 'Virginica']

