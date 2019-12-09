import pandas as pd
import numpy as np
import profile
from collections import defaultdict
from scipy.io.arff import loadarff


class DecisionTree:
    def __init__(self, max_depth=5, min_size=5):
        self.max_depth = max_depth
        self.min_size = min_size
        self.tree = None

    def best_split(self, X, y):
        best_attribute = None
        best_gini = float("inf")
        best_groups = None
        split_value = None

        for col_index in range(self.n_rows):
            for row in X:
                groups = self.test_split(col_index, row[col_index], X)
                gini = self.gini_index(groups, y)

                if gini < best_gini:
                    best_gini = gini
                    best_attribute = col_index
                    split_value = row[col_index]
                    best_groups = groups

        return {'leaf': False,
                'attribute': best_attribute,
                'split_value': split_value,
                'left': best_groups['left'],
                'right': best_groups['right']}

    def test_split(self, attribute, value, X):
        left = list()
        right = list()

        for index in range(len(X)):
            if X[index, attribute] < value:
                left.append(index)
            else:
                right.append(index)

        return {'left': left, 'right': right}

    def gini_index(self, groups, y):
        n_total = float(sum([len(group) for group in groups]))
        gini = 0.0

        for group in groups.values():
            size = len(group)
            score = 0.0
            if len(group) == 0:
                continue

            for class_value in self.classes:
                count = 0
                for item in group:
                    if y[item] == class_value:
                        count += 1
                p = count / size
                score += p * p

            gini += (1.0 - score) * (size / n_total)

        return gini

    def split(self, node, depth):
        right = node['right']
        left = node['left']

        # no split
        if not right or not left:
            node['left'] = self.leaf_node(left+right)
            node['right'] = self.leaf_node(left+right)
            return

        # is max depth
        if self.max_depth <= depth:
            node['right'] = self.leaf_node(right)
            node['left'] = self.leaf_node(left)
            return

        # check right child
        if len(right) <= self.min_size:
            node['right'] = self.leaf_node(right)
        else:

            node['right'] = self.best_split(self.X[right, :], self.y[right, :])
            self.split(node['right'], depth+1)

        # check left child
        if len(left) <= self.min_size:
            node['left'] = self.leaf_node(left)
        else:
            node['left'] = self.best_split(self.X[left, :], self.y[left, :])
            self.split(node['left'], depth+1)


    def leaf_node(self, indices):
        grouping = defaultdict()

        grouping['-1'] = [self.y[idx] for idx in indices].count(-1) / len(indices)
        grouping['1'] = [self.y[idx] for idx in indices].count(1) / len(indices)
        grouping['leaf'] = True

        return grouping

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.classes = set(int(item) for item in y)
        self.n_rows = len(X[0])

        root = self.best_split(X, y)
        self.split(root, 1)
        self.tree = root


# split into train, test, val
def preprocess_data(data):
    # remove 20% for testing
    data_copy = data.copy()
    testing_set = data_copy.sample(frac=0.2)
    data_copy = data_copy.drop(testing_set.index)

    # split remaining into 75% training 25% validation
    training_set = data_copy.sample(frac=0.75)
    validation_set = data_copy.drop(training_set.index)

    return training_set, testing_set, validation_set, data_copy


# Splitting into x and y
def x_y_split(dataset):
    dataset = dataset.to_numpy()
    n_rows = len(data[0])
    x = dataset[:len(dataset), :(n_rows - 1)].astype(int)
    y = dataset[:len(dataset), -1].astype(int)

    return x, y


if __name__ == '__main__':
    data, meta = loadarff("training.arff")
    phishing = pd.DataFrame(data)
    for column in phishing:
        phishing[column] = phishing[column].str.decode('utf-8')

    # I think -1 = Legitimate, 0 = Suspicious , and 1 = Phishing
    training_set: pd.DataFrame
    testing_set: pd.DataFrame
    validation_set: pd.DataFrame
    train_val_set: pd.DataFrame
    training_set, testing_set, validation_set, train_val_set = preprocess_data(phishing)

    # getting X and Y for testing, training, and validation
    x_test, y_test = x_y_split(testing_set)
    x_train, y_train = x_y_split(training_set)
    x_val, y_val = x_y_split(validation_set)
    x_train_val, y_train_val = x_y_split(train_val_set)

    tree = DecisionTree()
    # profile.run('tree.fit(x_train, y_train)')
    tree.fit(x_train, y_train)
