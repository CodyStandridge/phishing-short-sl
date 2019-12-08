from scipy.io.arff import loadarff
import pandas as pd
from collections import defaultdict


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
    y = dataset["Result"]
    x = dataset.drop(columns="Result")
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
    # x_test, y_test = x_y_split(testing_set)
    # x_train, y_train = x_y_split(training_set)
    # x_val, y_val = x_y_split(validation_set)
    # x_train_val, y_train_val = x_y_split(train_val_set)

    probs = defaultdict()
    num_legit = training_set.groupby("Result").size()[0]
    num_phish = training_set.groupby("Result").size()[1]
    print(num_legit)
    print(num_phish)

    for column in training_set:
        if column == "Result":
            continue
        probs[column] = defaultdict()
        probs[column]["phishing"] = {'yes': 0, 'no': 0, 'maybe': 0}
        probs[column]["legitimate"] = {'yes': 0, 'no': 0, 'maybe': 0}
        for index, row in training_set[column].items():
            if training_set["Result"][index] == '1':
                if row == '1':
                    probs[column]["phishing"]['yes'] += 1
                if row == '-1':
                    probs[column]["phishing"]['no'] += 1
                if row == '0':
                    probs[column]["phishing"]['maybe'] += 1
            else:
                if row == '1':
                    probs[column]["legitimate"]['yes'] += 1
                if row == '-1':
                    probs[column]["legitimate"]['no'] += 1
                if row == '0':
                    probs[column]["legitimate"]['maybe'] += 1

        probs[column]["phishing"] = {k: v / num_phish for k,v in probs[column]["phishing"].items()}
        probs[column]["legitimate"] = {k: v / num_legit for k,v in probs[column]["legitimate"].items()}

    # P (column = {yes, no, maybe} | {phishing, legitimate})
    for key in probs:
        print(key)
        print(probs[key])