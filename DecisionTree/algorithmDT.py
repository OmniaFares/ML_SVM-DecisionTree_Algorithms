import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None


class Node:

    def __init__(self, feature_index=None, left=None, right=None, value=None):
        self.feature_index = feature_index  # decision node
        self.left = left
        self.right = right
        self.value = value  # value of leaf node


class DecisionTree:

    def __init__(self):
        self.root = None
        self.depth = 0
        self.NumOfNodes = 0

    def build_tree(self, dataset, curr_depth=0):
        y = dataset[:, 0]
        num_features = 16
        self.NumOfNodes += 1
        best_split = self.get_best_split(dataset, num_features)
        if best_split['info_gain'] > 0:
            left_subtree = self.build_tree(best_split["dataset_left"], curr_depth + 1)
            right_subtree = self.build_tree(best_split["dataset_right"], curr_depth + 1)
            return Node(best_split["feature_index"], left_subtree, right_subtree)
        leaf_value = self.get_average(y)
        if curr_depth > self.depth:
            self.depth = curr_depth
        return Node(value=leaf_value)

    def get_average(self, y):
        democrat = 0
        republican = 0
        for item in y:
            if item == 'republican':
                republican += 1
            else:
                democrat += 1
        return ['democrat'] if democrat > republican else ['republican']

    def get_best_split(self, dataset, num_features):
        best_split = {"info_gain": 0}  # initial value
        max_info_gain = -1
        for feature_index in range(num_features):
            feature_index += 1  # to ignore the first column (output)
            dataset_left, dataset_right = self.split_dataset(dataset, feature_index)
            if len(dataset_left) > 0 and len(dataset_right) > 0:  # to check if the feature is redundant
                y, y_left, y_right = dataset[:, 0], dataset_left[:, 0], dataset_right[:, 0]
                curr_info_gain = self.information_gain(y, y_left, y_right)
                if curr_info_gain > max_info_gain:
                    best_split["feature_index"] = feature_index - 1  # to reset the index
                    best_split["dataset_left"] = dataset_left
                    best_split["dataset_right"] = dataset_right
                    best_split["info_gain"] = curr_info_gain
                    max_info_gain = curr_info_gain
        return best_split

    def split_dataset(self, dataset, feature_index):
        dataset_left = np.array([row for row in dataset if row[feature_index] == 'y'])
        dataset_right = np.array([row for row in dataset if row[feature_index] == 'n'])
        return dataset_left, dataset_right

    def information_gain(self, parent, child_left, child_right):
        s_left = len(child_left) / len(parent)
        s_right = len(child_right) / len(parent)
        info_gain = self.entropy(parent) - (s_left * self.entropy(child_left) + s_right * self.entropy(child_right))
        return info_gain

    def entropy(self, y):
        output_values = np.unique(y)
        entropy = 0
        for value in output_values:
            p = len(y[y == value]) / len(y)
            entropy += -p * np.log2(p)
        return entropy

    def predict(self, x):
        return [self.make_prediction(record, self.root) for record in x]

    def make_prediction(self, x, tree):
        if tree.value is not None:
            return tree.value
        feature_value = x[tree.feature_index]
        if feature_value == "y":
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)

    def start_dt(self, dataset):
        self.root = self.build_tree(dataset)
        return self.depth, self.NumOfNodes


def add_missing_values(x):
    for column in x:
        x.loc[x[column] == '?', column] = x[column].value_counts().idxmax()
    return x


def split_data(ratio, dataset):
    data_train = dataset.sample(frac=ratio)
    data_test = dataset.drop(data_train.index)

    x_train = data_train.iloc[:, data_train.columns != 'name'].values
    y_train = data_train.iloc[:, 0].values.reshape(-1, 1)

    x_test = data_test.iloc[:, data_test.columns != 'name'].values
    y_test = data_test.iloc[:, 0].values.reshape(-1, 1)

    return x_train, x_test, y_train, y_test


def perform(ratio, data):
    x_train, x_test, y_train, y_test = split_data(ratio, data)
    dataset = np.concatenate((y_train, x_train), axis=1)
    model = DecisionTree()
    depth, numberOfNodes = model.start_dt(dataset)
    y_predicated = model.predict(x_test)

    acc = np.sum(np.equal(y_test, y_predicated)) / len(y_predicated)
    return acc, depth, numberOfNodes


def point_one(file):
    ratio = 0.25
    print("Point 1:")
    file.write("\nPoint 1:")
    print("-------------------------------")
    file.write("\n-------------------------------")
    print(" Five different random experiments with ratio  = ", ratio)
    file.write("%s %s" % (" \nFive different random experiments with ratio  = ", ratio))
    for i in range(5):
        acc, depth, numberOfNodes = perform(ratio, data)
        print("  Depth tree = ", depth, ", Accuracy = ", acc * 100, ", number of nodes = ", numberOfNodes)
        file.write("%s %s %s %s %s %s" % ("  \nDepth tree = ", depth, ", Accuracy = ", acc * 100, ", number of nodes = ", numberOfNodes))


def point_two(file):
    print("\nPoint 2:")
    file.write("\n\n\nPoint 2:")
    print("-------------------------------")
    file.write("\n-------------------------------")
    ratio = 0.3
    for J in range(5):
        mean_depth, mean_acc, min_depth, min_acc, max_depth, max_acc, mean_num_of_nodes, max_num_of_nodes, min_num_of_nodes = 0, 0, 17, 1, -1, -1, 0, -1, 3000
        print(" Report of five different random experiments with ratio  = ", ratio)
        file.write("%s %s" % (" \n\nReport of five different random experiments with ratio  = ", ratio))
        for i in range(5):
            acc, depth, numberOfNodes = perform(ratio, data)
            if acc > max_acc:  max_acc = acc
            if acc < min_acc:  min_acc = acc
            if depth > max_depth:  max_depth = depth
            if depth < min_depth:  min_depth = depth
            if numberOfNodes > max_num_of_nodes:  max_num_of_nodes = numberOfNodes
            if numberOfNodes < min_num_of_nodes:  min_num_of_nodes = numberOfNodes
            mean_acc += acc
            mean_depth += depth
            mean_num_of_nodes += numberOfNodes
        ratio += 0.1

        mean_num_of_nodes /= 5
        mean_depth /= 5
        mean_acc /= 5

        print("  Mean Accuracy : ", mean_acc * 100)
        file.write("%s %s" % (" \nMean Accuracy : ", mean_acc * 100))
        print("  Max Accuracy : ", max_acc * 100)
        file.write("%s %s" % (" \nMax Accuracy : ", max_acc * 100))
        print("  Min Accuracy : ", min_acc * 100)
        file.write("%s %s" % (" \nMin Accuracy : ", min_acc * 100))
        print("  Mean Depth : ", mean_depth)
        file.write("%s %s" % (" \nMean Depth : ", mean_depth))
        print("  Max Depth : ", max_depth)
        file.write("%s %s" % (" \nMax Depth : ", max_depth))
        print("  Min Depth : ", min_depth)
        file.write("%s %s" % (" \nMin Depth : ", min_depth))
        print("  Mean number of nodes : ", mean_num_of_nodes)
        file.write("%s %s" % (" \nMean number of nodes : ", mean_num_of_nodes))
        print("  Max number of nodes : ", max_num_of_nodes)
        file.write("%s %s" % (" \nMax number of nodes : ", max_num_of_nodes))
        print("  Min number of nodes : ", min_num_of_nodes)
        file.write("%s %s" % (" \nMin number of nodes : ", min_num_of_nodes))


data = pd.read_csv('house-votes.csv')
data = add_missing_values(data)

file = open("DECISION TREE.txt", "w")
point_one(file)
point_two(file)
file.close()
