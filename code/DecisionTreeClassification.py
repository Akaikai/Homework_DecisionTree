from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from DecisionTree import classify, create_tree
from ModifiedDecisionTree import create_tree_modified

iris = load_iris()
x = iris.data.tolist()
y = iris.target.tolist()
attribute = ['sepal length', 'sepal width', 'petal length', 'petal width']
classification = ['Setosa', 'Versicolour', 'Virginica']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

decision_tree = create_tree(x_train, y_train)
decision_tree_modified = create_tree_modified(x_train, y_train, entropy_low=0.3)
print(decision_tree)
print(decision_tree_modified)

num_right1, num_right2 = 0, 0
for i in range (0, len(x_test)):
    predict1, predict2 = classify(decision_tree, x_test[i]), classify(decision_tree_modified, x_test[i])
    if predict1 == y_test[i]:
        num_right1 = num_right1 + 1
    if predict2 == y_test[i]:
        num_right2 = num_right2 + 1

accuracy1, accuracy2 = num_right1 / len(x_test), num_right2 / len(x_test)
print(accuracy1)
print(accuracy2)
