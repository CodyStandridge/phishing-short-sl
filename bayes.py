from scipy.io.arff import loadarff
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
from collections import defaultdict
from matplotlib import pyplot


# split into train, test, val
def train_test_split(data):
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
    y = dataset["Result"]
    x = dataset.drop(columns="Result")
    return x, y

class NaiveBayes:
    def fit(self, X, y):
        probs = defaultdict()
        num_legit = y.value_counts()[1]
        num_phish = y.value_counts()[0]
        for column in X:
            probs[column] = defaultdict()
            probs[column]["phishing"] = {'1': 0, '-1': 0, '0': 0}
            probs[column]["legitimate"] = {'1': 0, '-1': 0, '0': 0}
            for index, row in X[column].items():
                if y[index] == '1':
                    probs[column]["phishing"][row] += 1
                else:
                    probs[column]["legitimate"][row] += 1
            probs[column]["phishing"] = {k: v / num_phish for k, v in probs[column]["phishing"].items()}
            probs[column]["legitimate"] = {k: v / num_legit for k, v in probs[column]["legitimate"].items()}
            probs["phishing"] = num_phish / len(y.index)
            probs["legitimate"] = num_legit / len(y.index)
        self.probs = probs

    def predict_proba(self, X):
        # P (column = {yes, no, maybe} | {phishing, legitimate})
        predictions = []
        for _, row in X.iterrows():
            proba = {'legitimate': 1.0, 'phishing': 1.0}
            for column in X:
                proba['phishing'] *= self.probs[column]['phishing'][row[column]]
                proba['legitimate'] *= self.probs[column]['legitimate'][row[column]]
            proba['phishing'] *= self.probs['phishing']
            proba['legitimate'] *= self.probs['legitimate']
            predictions.append(proba)
        return predictions

    def predict(self, X):
        predictions = []
        pred = self.predict_proba(X)
        for proba in pred:
            prob_phish = proba['phishing']
            prob_legit = proba['legitimate']
            if prob_legit > prob_phish:
                predictions.append(-1)
            else:
                predictions.append(1)
        return predictions

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
    training_set, testing_set, validation_set, train_val_set = train_test_split(phishing)

    # getting X and Y for testing, training, and validation
    x_test, y_test = x_y_split(testing_set)
    x_train, y_train = x_y_split(training_set)
    x_val, y_val = x_y_split(validation_set)
    x_train_val, y_train_val = x_y_split(train_val_set)

    bayes = NaiveBayes()
    nb = GaussianNB()
    nb.fit(x_train.to_numpy().astype(int), y_train.to_numpy().astype(int))
    bayes.fit(x_train, y_train)

    predictions = bayes.predict(x_test)
    # predictions = nb.predict(x_test.to_numpy().astype(int))
    # predict_probas = nb.predict_proba(x_test.to_numpy().astype(int))
    predict_probas = bayes.predict_proba(x_test)
    y_test = y_test.to_numpy().astype(int)

    # find a, b, c, and d for tree, forest, and boost
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    for i in range(len(y_test)):
        if predictions[i] == y_test[i]:
            if predictions[i] == 1:
                true_pos += 1
            else:
                true_neg += 1
        else:
            if predictions[i] == 1:
                false_pos += 1
            else:
                false_neg += 1

    print("True Pos: {}, True Neg: {}, False Pos: {}, False Neg: {}".format(true_pos, true_neg, false_pos, false_neg))
    pod = true_pos / (true_pos + false_neg)
    pofd = false_pos / (false_pos + true_neg)
    print("POD: {}. POFD: {}".format(pod, pofd))
    print("CSI: {}".format(true_pos / (true_pos + false_pos + false_neg)))

    #ROC Curve
    thresh = np.linspace(0, 1, 1001)
    pod_bayes = []
    pofd_bayes = []

    for crit_thresh in thresh:
        p_bayes = []
        for probs in predict_probas:
            if probs['phishing'] >= crit_thresh:
                p_bayes.append(1)
            else:
                p_bayes.append(0)
        a = 0
        b = 0
        c = 0
        d = 0

        for i in range(len(y_test)):
            if p_bayes[i] == y_test[i]:
                if p_bayes[i] == 1:
                    a += 1
                else:
                    d += 1
            else:
                if p_bayes[i] == 1:
                    b += 1
                else:
                    c += 1
        pod_bayes.append(a / (a + c) if (a + c) != 0 else 0)
        pofd_bayes.append(b / (b + d) if (b + d) != 0 else 0)

    # Plot Forest
    pyplot.xlabel("POFD Bayes")
    pyplot.ylabel("POD Bayes")
    pyplot.title("ROC Curve Bayes")
    pyplot.plot(pofd_bayes, pod_bayes)
    pyplot.plot(thresh, thresh, "--")
    pyplot.show()