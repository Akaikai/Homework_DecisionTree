import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from DecisionTree import classify, create_tree, get_depth
from ModifiedDecisionTree import create_tree_modified

import SoftDecisionTree


iris = load_iris()
x = iris.data.tolist()
y = iris.target.tolist()
attribute = ['sepal length', 'sepal width', 'petal length', 'petal width']
classification = ['Setosa', 'Versicolour', 'Virginica']


def plot_new_entropy_accuracy():
    accuracy_values1, accuracy_values2 = [], []
    depth_values1, depth_values2 = [], []
    for entropy in range(0, 40, 2):
        total_right1, total_right2 = 0, 0
        total_depth1, total_depth2 = 0, 0
        for i in range(0, 100):
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
            decision_tree = create_tree(x_train, y_train)
            decision_tree_modified = create_tree_modified(x_train, y_train, entropy_low=entropy / 100)

            total_depth1, total_depth2 = total_depth1 + get_depth(decision_tree), total_depth2 + get_depth(decision_tree_modified)
            num_right1, num_right2 = 0, 0
            for i in range(0, len(x_test)):
                predict1, predict2 = classify(decision_tree, x_test[i]), classify(decision_tree_modified, x_test[i])
                if predict1 == y_test[i]:
                    num_right1 = num_right1 + 1
                if predict2 == y_test[i]:
                    num_right2 = num_right2 + 1
            total_right1, total_right2 = total_right1 + num_right1, total_right2 + num_right2
        accuracy1, accuracy2 = total_right1 / (len(y) * 0.2 * 100), total_right2 / (len(y) * 0.2 * 100)
        average_depth1, average_depth2 = total_depth1 / 100, total_depth2 / 100
        print(accuracy1, accuracy2)
        accuracy_values1.append(accuracy1)
        accuracy_values2.append(accuracy2)
        depth_values1.append(average_depth1)
        depth_values2.append(average_depth2)

    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111)
    plt.title('The relationship between smallest acceptable new entropy and accuracy / max depth')
    plt.xlabel('Smallest acceptable new entropy')
    ax1.plot(np.linspace(0, 0.40, 20), accuracy_values1, linestyle=':', marker='o', label='Classical-Accuracy')
    ax1.plot(np.linspace(0, 0.40, 20), accuracy_values2, linestyle=':', marker='o', label='Modified-Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax2 = ax1.twinx()
    ax2.plot(np.linspace(0, 0.40, 20), depth_values1, 'green', linestyle='-', marker='o',  label='Classical-MaxDepth')
    ax2.plot(np.linspace(0, 0.40, 20), depth_values2, 'red', linestyle='-', marker='o', label='Modified-MaxDepth')
    ax2.set_ylabel('Max Depth of the Decision Tree')
    ax2.set_ylim(0, 10)
    ax2.legend()
    plt.show()


def plot_n_accuracy():
    accuracy_values1, accuracy_values2 = [], []
    n_values = []
    for n in range(1, 40):
        total_right1, total_right2 = 0, 0
        for epoch in range(0, 100):
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
            decision_tree = create_tree(x_train, y_train)

            num_right1, num_right2 = 0, 0
            for i in range(0, len(x_test)):
                predict1, predict2 = SoftDecisionTree.classify(decision_tree, x_test[i], n * 1), classify(decision_tree, x_test[i])
                if predict1 == y_test[i]:
                    num_right1 = num_right1 + 1
                if predict2 == y_test[i]:
                    num_right2 = num_right2 + 1

            total_right1, total_right2 = total_right1 + num_right1, total_right2 + num_right2
        accuracy1, accuracy2 = total_right1 / (len(y) * 0.2 * 100), total_right2 / (len(y) * 0.2 * 100)
        accuracy_values1.append(accuracy1)
        accuracy_values2.append(accuracy2)
        n_values.append(n)
        print(accuracy1, accuracy2)

    plt.figure(figsize=(8, 5))
    plt.title('n-accuracy relationship')
    plt.xlabel('n')
    plt.ylabel('accuracy')
    plt.plot(n_values, accuracy_values1, linestyle=':', marker='.', label='Soft Boundary')
    plt.plot(n_values, accuracy_values2, linestyle=':', marker='.', label='Hard Boundary')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    # plot_new_entropy_accuracy()
    plot_n_accuracy()