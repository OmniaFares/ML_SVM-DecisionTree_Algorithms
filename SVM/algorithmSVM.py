import itertools
import pandas as pd
import numpy as np
import random

FullData = pd.read_csv("heart.csv")


def split_data(fulldata):
    shuffle_df = fulldata.sample(frac=1)  # return a sample random of x's (frac 1 means all of data 100% from it)
    size_of_train = int(0.80 * len(fulldata))
    train_data = shuffle_df[:size_of_train]
    test_data = shuffle_df[size_of_train:]
    return train_data, test_data


def Normalization(dataset):
    normalized = (dataset - dataset.mean()) / dataset.std()
    return normalized


def convert_y(y):
    return np.where(y <= 0, -1, 1)


def prediction(w, x, b):
    h = np.dot(x, w.T) + b
    for i in range(len(h)):
        if h[i] > 0:
            h[i] = 1
        elif h[i] < 0:
            h[i] = -1
    return h


def hypothesis(x, w, b):
    return np.dot(x, w.T) + b


def Accuracy(y_predicted, y_actual):
    total = 0
    for i in range(len(y_actual)):
        if y_actual[i] == y_predicted[i]:
            total += 1
    total_accuracy = total / len(y_actual)
    return total_accuracy


def gradient_descent(lamda, w, b, alpha, x, y, iterations):
    for i in range(iterations):
        for j in range(len(x)):
            h = hypothesis(x[j], w, b)
            Condition = y[j] * h
            if Condition >= 1:
                w = w - (2 * w * alpha * lamda)
            else:
                w = w + alpha * ((x[j] * y[j]) - (2 * lamda * w))
                b = b + (alpha * y[j])
    return w, b


def find_different_features(dataset):
    combinations = []
    for i in range(1, 14):
        new_features_group = list(itertools.combinations(dataset, i))
        combinations.append(new_features_group)
    features = []
    for i in combinations:
        for j in i:
            features.append(list(j))
    return features


FullData['target'] = convert_y(FullData['target'])
Train_data, Test_data = split_data(FullData)

Y_train = np.array(Train_data['target']).flatten()
Y_test = np.array(Test_data['target']).flatten()

X = find_different_features(FullData.drop(['target'], axis=1))
random.shuffle(X)

lamda = 0.001
b = 0
iterations = 800
alphas = [0.001, 0.01, 0.03, 0.1, 0.3]

# start
for a in alphas:
    Arr_of_acc = []
    features = []

    for i in range(15):
        X_train = Train_data[X[i]]
        X_test = Test_data[X[i]]

        X_train = np.array(Normalization(X_train))
        X_test = np.array(Normalization(X_test))

        w = np.zeros(np.size(X_train, 1))

        (w, b) = gradient_descent(lamda, w, b, a, X_train, Y_train, iterations)
        y_predict = prediction(w, X_test, b)
        Acc = Accuracy(y_predict, Y_test)

        Arr_of_acc.append(Acc)
        features.append(X[i])

    max_acc = max(Arr_of_acc)
    index = Arr_of_acc.index(max_acc)
    print("###############################################################")
    print("for alpha = ", a)
    print("best features found:  ", features[index], "\n with accuracy = ", max_acc)