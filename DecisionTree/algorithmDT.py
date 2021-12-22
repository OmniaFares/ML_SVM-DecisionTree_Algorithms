import pandas as pd
import numpy as np

class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.value = value

class DecisionTree():
    def __init__(self):
        self.root = None

    def build_tree(self, dataset):
        X, Y = dataset[:, 1:], dataset[:, 0]
        num_samples, num_features = np.shape(X)
        best_split = self.get_best_split(dataset, num_features)
        if("info_gain" in best_split):
            if best_split['info_gain'] > 0:
                left_subtree = self.build_tree(best_split["dataset_left"])
                right_subtree = self.build_tree(best_split["dataset_right"])
                return Node(best_split["feature_index"], best_split["threshold"],
                            left_subtree, right_subtree, best_split["info_gain"])
        leaf_value = np.unique(Y)
        return Node(value=leaf_value)

    def get_best_split(self, dataset, num_features):
        best_split = {}
        max_info_gain = -float("inf")
        for feature_index in range(num_features):
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

    def print_tree(self, tree=None, indent=" "):
        if not tree:
            tree = self.root
        if tree.value is not None:
            print(tree.value)
        else:
            print("X_" + str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)

    def predict(self, X):
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions

    def make_prediction(self, x, tree):
        if tree.value != None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val == tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)

    def fit(self, X, Y):
        # dataset = np.concatenate((X, Y), axis=1)
        dataset = np.hstack((X,np.array([Y]).T))
        self.root = self.build_tree(dataset)


pd.options.mode.chained_assignment = None


def add_missing_values(x):
    for column in x:
        x.loc[x[column] == '?', column] = x[column].value_counts().idxmax()
    return x


def split_data(ratio, x, y):
    x_train = x.sample(frac = ratio)
    x_test = x.drop(x_train.index)
    y_train = y.sample(frac = ratio)
    y_test = y.drop(y_train.index)
    return x_train, x_test, y_train, y_test


data = pd.read_csv('house-votes.csv')
x = data.iloc[:, data.columns != 'name']
x = add_missing_values(x)
y = data.iloc[:, 0]
ratio = 0.25
x_train, x_test, y_train, y_test = split_data(ratio, x, y)


classifier = DecisionTree()
classifier.fit(x_train,y_train)
classifier.print_tree()

Y_pred = classifier.predict(x_test)
acc = np.sum(np.equal(y_test, Y_pred)) / len(Y_pred)

print(acc)