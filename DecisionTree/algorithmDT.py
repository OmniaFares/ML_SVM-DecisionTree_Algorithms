
import pandas as pd
import matplotlib.pyplot as plt

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