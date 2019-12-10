import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.io.arff import loadarff


class DecisionTree:
    def __init__(self, max_depth=5, min_size=5):
        self.max_depth = max_depth
        self.min_size = min_size
        self.tree = None

    def best_split(self, indices):
        best_attribute = None
        best_gini = float("inf")
        best_groups = None
        split_value = None
        columns = random.sample(range(self.n_cols), int(np.sqrt(self.n_cols)))

        for col_index in columns:
            temp_x = set([self.X[item, col_index] for item in indices])
            for item_value in temp_x:
                groups = self.test_split(col_index, item_value, indices)
                gini = self.gini_index(groups)

                if gini < best_gini:
                    best_gini = gini
                    best_attribute = col_index
                    split_value = item_value
                    best_groups = groups

        return {'leaf': False,
                'attribute': best_attribute,
                'split_value': split_value,
                'left': best_groups['left'],
                'right': best_groups['right']}

    def test_split(self, attribute, value, indices):
        left = []
        right = []

        for index in indices:
            if self.X[index, attribute] < value:
                left.append(index)
            else:
                right.append(index)

        return {'left': list(left), 'right': list(right)}

    def gini_index(self, groups):
        n_total = float(sum([len(group) for group in groups.values()]))
        gini = 0.0

        for group in groups.values():
            size = float(len(group))
            score = 0.0
            if size == 0:
                continue

            for class_value in self.classes:
                p = [self.y[index] for index in group].count(class_value) / size
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
            node['right'] = self.best_split(right)
            self.split(node['right'], depth+1)

        # check left child
        if len(left) <= self.min_size:
            node['left'] = self.leaf_node(left)
        else:
            node['left'] = self.best_split(left)
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
        self.classes = set(int(item) for item in self.y)
        self.n_cols = len(self.X[0])
        root = self.best_split(range(len(self.X)))
        self.split(root, 1)
        self.tree = root

    def predict_tree_helper(self, row, node):
        if node['leaf']:
            return node
        if row[node['attribute']] >= node['split_value']:
            return self.predict_tree_helper(row, node['right'])
        else:
            return self.predict_tree_helper(row, node['left'])

    def predict_proba(self, dataset):
        preds = []
        for i in range(len(dataset)):
            preds.append(self.predict_tree_helper(dataset[i], self.tree))
        return preds

    def predict(self, dataset):
        final_predictions = []

        for index in range(len(dataset)):
            pred = self.predict_tree_helper(dataset[index], self.tree)
            best = -1
            best_key = None

            for k, v in pred.items():
                if v > best and k != 'leaf':
                    best = v
                    best_key = k

            final_predictions.append(best_key)

        return final_predictions


# split into train, test, val
def preprocess_data(dataset):
    # remove 20% for testing
    data_copy = dataset.copy()
    testing_set = data_copy.sample(frac=0.2)
    data_copy = data_copy.drop(testing_set.index)

    # split remaining into 75% training 25% validation
    training_set = data_copy.sample(frac=0.75)
    validation_set = data_copy.drop(training_set.index)

    return training_set, testing_set, validation_set, data_copy


# Splitting into x and y
def x_y_split(dataset):
    dataset = dataset.to_numpy()
    n_rows = len(dataset[0])
    x = dataset[:len(dataset), :(n_rows - 1)].astype(int)
    y = dataset[:len(dataset), -1].astype(int)

    return x, y


# calculate the accuracy of the model
def stats(X, y, preds):
    correct = 0
    for index in range(len(X)):
        if int(preds[index]) == y[index]:
            correct += 1

    forest_pod_pofd_csi = pod_pofd_csi(preds, y)

    print("Accuracy: " + str(correct/len(y)))
    print("POD: " + str(forest_pod_pofd_csi[0]))
    print("POFD: " + str(forest_pod_pofd_csi[1]))
    print("CSI: " + str(forest_pod_pofd_csi[2]))


# determinize
def determinize(thres, prob):
    if prob >= thres:
        return 1
    else:
        return -1


# probability of detection and false detection
def pod_pofd_csi(pred, obs):
    a = 0   # true pos
    c = 0   # false pos
    b = 0   # false neg
    d = 0   # true neg

    for index in range(len(pred)):
        if int(pred[index]) == obs[index] == 1:
            a += 1
        elif int(pred[index]) == 1 and obs[index] == -1:
            b += 1
        elif int(pred[index]) == -1 and obs[index] == 1:
            c += 1
        elif int(pred[index]) == obs[index] == -1:
            d += 1

    pod = a / (a + c)
    pofd = b / (b + d)
    csi = a / (a + c + b)
    return [pod, pofd, csi]


# claculate the ROC curve
def roc_curve(tree_preds, y):
    thresholds = np.arange(start=0.000, stop=1.001, step=0.001)
    tree_pod_pofd_csi = []
    tree_det_pred = []
    for index in range(len(thresholds) - 1):
        tree_det_pred.append([determinize(thresholds[index], n['1']) for n in tree_preds])
        tree_pod_pofd_csi.append(pod_pofd_csi(tree_det_pred[index], y))
    tree_pod_pofd_csi.append([0, 0])

    # area under the curve
    auc(list(zip(*tree_pod_pofd_csi))[1], list(zip(*tree_pod_pofd_csi))[0])

    # plot the ROC curve
    plt.ylabel('POD')
    plt.xlabel('POFD')
    plt.title('ROC Curve (Decision Tree)')
    plt.axis([0, 1, 0, 1])
    plt.plot(thresholds, thresholds, 'b--',
             list(zip(*tree_pod_pofd_csi))[1], list(zip(*tree_pod_pofd_csi))[0], 'r-')
    plt.show()


# calculates the area under the curve
def auc(x_axis, y_axis):

    areas = []
    for idx in range(len(x_axis) - 1):
        # trapezoid formula
        ys = y_axis[idx] + y_axis[idx + 1]
        xs = x_axis[idx + 1] - x_axis[idx]
        areas.append(1/2 * ys * xs)

    print("AUC: " + str(abs(sum(areas))))


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

    # create and train the model
    tree = DecisionTree()
    tree.fit(x_train, y_train)

    # predictions, accuracy, roc curve
    val_preds = tree.predict(x_val)
    val_pred_proba = tree.predict_proba(x_val)
    stats(x_val, y_val, val_preds)
    roc_curve(val_pred_proba, y_val)

