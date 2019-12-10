import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import mode
from collections import defaultdict
from scipy.io.arff import loadarff


class RandomForest:
    def __init__(self, num_trees=5, max_depth=5, min_size=5):
        self.max_depth = max_depth
        self.min_size = min_size
        self.num_trees = num_trees
        self.tree = []

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
        for index in range(self.num_trees):
            indices = random.choices(range(len(X)), k=len(X))
            self.X = X[indices]
            self.y = y[indices]
            self.classes = set(int(item) for item in self.y)
            self.n_cols = len(self.X[0])
            root = self.best_split(range(len(self.X)))
            self.split(root, 1)
            self.tree.append(root)

    def predict_tree_helper(self, row, node):
        if node['leaf']:
            return node
        if row[node['attribute']] >= node['split_value']:
            return self.predict_tree_helper(row, node['right'])
        else:
            return self.predict_tree_helper(row, node['left'])

    def predict_tree(self, row):
        predictions = {}
        temp_pred = []
        for index in range(self.num_trees):
            temp_pred.append(self.predict_tree_helper(row, self.tree[index]))

        for p in temp_pred:
            total = 0
            for k, v in p.items():
                if v != 'leaf':
                    if k not in predictions:
                        predictions[k] = v
                        total += v
                    else:
                        predictions[k] += v
                        total += v
            for k, v in predictions.items():
                predictions[k] = v / total

        return predictions

    def predict(self, dataset):
        final_predictions = []
        final_pred_proba = []
        for trees in range(self.num_trees):
            temp_pred = []
            temp_pred_proba = []
            for index in range(len(dataset)):
                pred = self.predict_tree(dataset[index])
                best = -1
                best_key = None

                for k, v in pred.items():
                    if v > best and k != 'leaf':
                        best = v
                        best_key = k

                temp_pred.append(best_key)
                temp_pred_proba.append(pred['1'])

            final_predictions.append(temp_pred)
            final_pred_proba.append(temp_pred_proba)

        val, count = mode(final_predictions, axis=0)
        val1 = np.mean(final_pred_proba, axis=0)

        return val.ravel().tolist(), val1.ravel().tolist()


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


# calculate the accuracy of the model
def stats(X, y, preds, trees):
    for i in range(len(trees)):
        correct = 0
        print("STATS FOR FOREST WITH: " + str(trees[i]) + " trees")
        for index in range(len(X)):
            if int(preds[i][index]) == y[index]:
                correct += 1

        forest_pod_pofd_csi = pod_pofd_csi(preds[i], y)

        print("Accuracy: " + str(correct/len(y)))
        print("POD: " + str(forest_pod_pofd_csi[0]))
        print("POFD: " + str(forest_pod_pofd_csi[1]))
        print("CSI: " + str(forest_pod_pofd_csi[2]) + '\n')


# claculate the ROC curve
def roc_curve(forest_preds, y, n_trees):
    thresholds = np.arange(start=0.000, stop=1.001, step=0.001)

    forest_pod_pofd_csi = []
    forest_det_pred = []
    for i in range(len(n_trees)):
        temp_pod_pofd_csi = []
        temp_det_pred = []
        for index in range(len(thresholds) - 1):
            temp_det_pred.append([determinize(thresholds[index], f) for f in forest_preds[i]])
            temp_pod_pofd_csi.append(pod_pofd_csi(temp_det_pred[index], y))
        temp_pod_pofd_csi.append([0, 0])
        forest_pod_pofd_csi.append(temp_pod_pofd_csi)
        forest_det_pred.append(temp_det_pred)

    # area under the curve
    for i in range(len(n_trees)):
        print("AUC for Forest with " + str(n_trees[i]) + " trees")
        auc(list(zip(*forest_pod_pofd_csi[i]))[1], list(zip(*forest_pod_pofd_csi[i]))[0])

    # plot the ROC curve
    plt.ylabel('POD')
    plt.xlabel('POFD')
    plt.title('ROC Curve (Random Forest)')
    plt.axis([0, 1, 0, 1])
    plt.plot(thresholds, thresholds, 'b--')

    # plot the different ROC curves
    for i in range(len(n_trees)):
        plt.plot(list(zip(*forest_pod_pofd_csi[i]))[1], list(zip(*forest_pod_pofd_csi[i]))[0], label=str(n_trees[i]))
    plt.legend(title="number of trees")
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

    # create and train the models
    n_trees = [5, 20, 50, 100]
    forests = []
    val_preds = []
    val_pred_probas = []
    for n in range(len(n_trees)):
        forests.append(RandomForest(num_trees=n_trees[n], max_depth=10))
        forests[n].fit(x_train, y_train)
        pred, proba = forests[n].predict(x_val)
        val_preds.append(pred)
        val_pred_probas.append(proba)

    # predictions, accuracy, roc curve
    stats(x_val, y_val, val_preds, n_trees)
    roc_curve(val_pred_probas, y_val, n_trees)

