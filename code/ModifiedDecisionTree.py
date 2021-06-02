from DecisionTree import get_dataset, get_shannon_ent, get_threshold, split_dataset


def get_best_feature_threshold(x, y):  # to find a feature and threshold to get the maximal information entropy gain
    num_features = len(x[0])  # numbers of features
    base_entropy = get_shannon_ent(x, y)
    best_info_gain = 0
    best_feature = -1
    best_value = 0
    best_new_entropy = 0
    for i in range(num_features):
        feature_list = [example[i] for example in x]
        thres = get_threshold(feature_list)
        for value in thres:
            sub_data1, sub_label1, sub_data2, sub_label2 = split_dataset(x, y, i, value)
            new_entropy = len(sub_data1) / (len(sub_data1) + len(sub_data2)) * get_shannon_ent(sub_data1, sub_label1) + len(sub_data2) / (len(sub_data1) + len(sub_data2)) * get_shannon_ent(sub_data2, sub_label2)
            info_gain = base_entropy - new_entropy
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = i
                best_value = value
                best_new_entropy = new_entropy
    return best_feature, best_value, best_new_entropy


def create_tree_modified(x, y, entropy_low):
    if y.count(y[0]) == len(y):
        return classification[y[0]]
    best_feat, best_val, best_entropy = get_best_feature_threshold(x, y)
    if best_entropy < entropy_low:
        return classification[max(y, key=y.count)]  # return majority label of the dataset
    my_tree = {attribute[best_feat]: {}}  # answer is saved through dictionary
    sub_data1, sub_label1, sub_data2, sub_label2 = split_dataset(x, y, best_feat, best_val)
    key_sub1, key_sub2 = str('<') + str(round(best_val, 2)), str('>') + str(round(best_val, 2))
    # iteration
    if len(sub_data1) > 0:
        my_tree[attribute[best_feat]][key_sub1] = create_tree_modified(sub_data1, sub_label1, entropy_low)
    if len(sub_data2) > 0:
        my_tree[attribute[best_feat]][key_sub2] = create_tree_modified(sub_data2, sub_label2, entropy_low)
    return my_tree


attribute = ['sepal length', 'sepal width', 'petal length', 'petal width']
classification = ['Setosa', 'Versicolour', 'Virginica']


if __name__ == '__main__':
    data, label = get_dataset()
    print(create_tree_modified(data, label, entropy_low=0.12))
