from sklearn.datasets import load_iris
import math


def get_dataset():  # create dataset
    iris = load_iris()
    x = iris.data
    y = iris.target
    return x.tolist(), y.tolist()


def neg_log(x):  # to get the negative log
    if x == 0:
        return 0
    else:
        return - x * math.log(x, 2)


def get_shannon_ent(x, y):  # calculate shannon entropy
    length = len(x)  # rows of the dataset
    label_count = {}  # numbers of each label
    for idx in range(0, length):
        if y[idx] not in label_count.keys():
            label_count[y[idx]] = 0
        label_count[y[idx]] = label_count[y[idx]] + 1

    shannon_ent = 0
    for key in label_count:
        probability = label_count[key] / length
        shannon_ent = shannon_ent + neg_log(probability)
    return shannon_ent


def get_threshold(input):  # turn continuous variable into discrete variable
    input.sort()
    output = []
    for i in range(0, len(input) - 1):
        output.append((input[i] + input[i + 1]) / 2)
    return output


def split_dataset(x, y, feature, value):  # split the dataset into two parts based on the value of certain feature
    sub_data1, sub_data2 = [], []
    sub_label1, sub_label2 = [], []
    for i in range(0, len(x)):
        if x[i][feature] < value:
            sub_data1.append(x[i])
            sub_label1.append(y[i])
        else:
            sub_data2.append(x[i])
            sub_label2.append(y[i])
    return sub_data1, sub_label1, sub_data2, sub_label2


def get_best_feature_threshold(x, y):  # to find a feature and threshold to get the maximal information entropy gain
    num_features = len(x[0])  # numbers of features
    base_entropy = get_shannon_ent(x, y)
    best_info_gain = 0
    best_feature = -1
    best_value = 0
    for i in range(num_features):  # find answer in each attribute
        feature_list = [example[i] for example in x]
        thres = get_threshold(feature_list)
        for value in thres: # find value in each dividing value
            sub_data1, sub_label1, sub_data2, sub_label2 = split_dataset(x, y, i, value)
            new_entropy = len(sub_data1) / (len(sub_data1) + len(sub_data2)) * get_shannon_ent(sub_data1, sub_label1) + len(sub_data2) / (len(sub_data1) + len(sub_data2)) * get_shannon_ent(sub_data2, sub_label2)
            info_gain = base_entropy - new_entropy
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = i
                best_value = value
    return best_feature, best_value


def create_tree(x, y):
    if y.count(y[0]) == len(y):
        return classification[y[0]]
    best_feat, best_val = get_best_feature_threshold(x, y)
    my_tree = {attribute[best_feat]: {}}  # answer is saved through dictionary
    sub_data1, sub_label1, sub_data2, sub_label2 = split_dataset(x, y, best_feat, best_val)
    key_sub1, key_sub2 = str('<') + str(round(best_val, 2)), str('>') + str(round(best_val, 2))
    if len(sub_data1) > 0:
        my_tree[attribute[best_feat]][key_sub1] = create_tree(sub_data1, sub_label1)
    if len(sub_data2) > 0:
        my_tree[attribute[best_feat]][key_sub2] = create_tree(sub_data2, sub_label2)
    return my_tree


def get_depth(input_tree):
    max_depth = 0
    first_string = next(iter(input_tree))
    second_dict = input_tree[first_string]  # get next dictionary
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            this_depth = 1 + get_depth((second_dict[key]))
        else:
            this_depth = 1
        if this_depth > max_depth:  # update depth
            max_depth = this_depth
    return max_depth


def classify(input_tree, test_data):
    first_string = next(iter(input_tree))
    second_dict = input_tree[first_string]
    feature_index = attribute.index(first_string)
    global class_label
    for key in second_dict.keys():
        if test_data[feature_index] < eval(key[1:]) and key[0] == '<':
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key], test_data)  # if value is dict type, then go on
            else:
                class_label = classification.index(second_dict[key])
        elif test_data[feature_index] > eval(key[1:]) and key[0] == '>':
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key], test_data)
            else:
                class_label = classification.index(second_dict[key])
    return class_label


attribute = ['sepal length', 'sepal width', 'petal length', 'petal width']
classification = ['Setosa', 'Versicolour', 'Virginica']


if __name__ == '__main__':
    data, label = get_dataset()
    print(create_tree(data, label))
