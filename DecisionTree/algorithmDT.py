import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None

class Node():

    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.value = value

class DecisionTree():

    def __init__(self, max_depth = 16):
        self.root = None
        self.depth = 0
        self.max_depth = max_depth

    def build_tree(self, dataset, curr_depth=0):
        X, Y = dataset[:, 1:], dataset[:, 0]
        num_samples, num_features = np.shape(X)
        best_split = self.get_best_split(dataset, num_features)
        if best_split['info_gain'] > 0 and curr_depth < self.max_depth:
            left_subtree = self.build_tree(best_split["dataset_left"], curr_depth + 1)
            right_subtree = self.build_tree(best_split["dataset_right"], curr_depth + 1)
            return Node(best_split["feature_index"], best_split["threshold"],
                        left_subtree, right_subtree, best_split["info_gain"])

        leaf_value = self.get_average(Y)
        if curr_depth > self.depth:
            self.depth = curr_depth
        return Node(value=leaf_value)

    def get_average(self, Y):
        democrat = 0
        republican = 0
        for item in Y:
            if item == 'republican':
                republican += 1
            else:
                democrat += 1
        return ['democrat'] if democrat > republican else ['republican']

    def get_best_split(self, dataset, num_features):
        best_split = {}
        best_split["info_gain"] = 0
        max_info_gain = -float("inf")
        for feature_index in range(num_features): #0-15 after add one 1-16
            feature_index += 1
            dataset_left, dataset_right = self.split(dataset, feature_index)
            if len(dataset_left) > 0 and len(dataset_right) > 0:
                y, left_y, right_y = dataset[:, 0], dataset_left[:, 0], dataset_right[:, 0]
                curr_info_gain = self.information_gain(y, left_y, right_y)
                if curr_info_gain > max_info_gain:
                    best_split["feature_index"] = feature_index
                    best_split["threshold"] = "y"
                    best_split["dataset_left"] = dataset_left
                    best_split["dataset_right"] = dataset_right
                    best_split["info_gain"] = curr_info_gain
                    max_info_gain = curr_info_gain
        return best_split

    def split(self, dataset, feature_index):
        dataset_left = np.array([row for row in dataset if row[feature_index] == 'y'])
        dataset_right = np.array([row for row in dataset if row[feature_index] == 'n'])
        return dataset_left, dataset_right

    def information_gain(self, parent, l_child, r_child):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        gain = self.entropy(parent) - (weight_l * self.entropy(l_child) + weight_r * self.entropy(r_child))
        return gain

    def entropy(self, y):
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy

    def predict(self, X):
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions

    def make_prediction(self, x, tree):
        if tree.value != None: return tree.value
        feature_val = x[tree.feature_index - 1]
        if feature_val == tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)

    def fit(self, X, Y):
        dataset = np.concatenate((Y, X), axis=1)
        self.root = self.build_tree(dataset)
        return self.depth

def add_missing_values(x):
    for column in x:
        x.loc[x[column] == '?', column] = x[column].value_counts().idxmax()
    return x


def split_data(ratio, x, y):
    x_train = x.sample(frac = ratio)
    x_test = x.drop(x_train.index).values
    x_train = x_train.values

    y_train = y.sample(frac = ratio)
    y_test = y.drop(y_train.index).values.reshape(-1, 1)
    y_train = y_train.values.reshape(-1, 1)

    return x_train, x_test, y_train, y_test

def perform(ratio, x, y):
    x_train, x_test, y_train, y_test = split_data(ratio, x, y)

    classifier = DecisionTree()
    depth = classifier.fit(x_train, y_train)
    Y_pred = classifier.predict(x_test)

    acc = np.sum(np.equal(y_test, Y_pred)) / len(Y_pred)
    return acc, depth

data = pd.read_csv('house-votes.csv')
x = data.iloc[:, data.columns != 'name']
x = add_missing_values(x)
y = data.iloc[:, 0]

ratio = 0.25
print("point 1:")
for i in range(5):
    acc, depth = perform(ratio, x, y)
    print("depth tree = ", depth, "accuracy = ", acc)

print("point 2:")
print("ratio in range (30-70%)")
ratio = 0.3
mean_depth = 0
mean_acc = 0
min_depth = 17
min_acc = 1
max_depth = -1
max_acc = -1
for i in range(5):
    acc, depth = perform(ratio, x, y)
    if acc > max_acc:
        max_acc = acc
    if acc < min_acc:
        min_acc = acc
    if depth > max_depth:
        max_depth = depth
    if depth < min_depth:
        min_depth = depth
    mean_acc += acc
    mean_depth += depth
    ratio += 0.1

mean_depth /= 5
mean_acc /= 5

print("Mean Accuracy : ", mean_acc)
print("Max Accuracy : ", max_acc)
print("Min Accuracy : ", min_acc)
print("Mean Depth : ", mean_depth)
print("Max Depth : ", max_depth)
print("Min Depth : ", min_depth)
