"""
    Train the Rwguard process monitor component based on Random Forests
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
from sklearn import metrics
import csv
import pandas as pd

"""
Load the train and testing dataset. Pay attention you have them already split up in two files!
You can load the data either using the csv library or pandas whichever you prefer
The feature names are in this order Read, write, open, close, fast read, fast write, fast close, fast open, label - since they are in different files you do not even need the label you can build the targets (train and test_y) using numerical values like 0 for one class and 1 for the other
"""

df_benign_train = pd.read_csv("./dataset/benign/benign_rwguard_features_3sec.csv", header=None)
benign_train_x = df_benign_train.iloc[:, :-1].values
benign_train_y = [0] * len(benign_train_x)


df_ransomware_train = pd.read_csv("./dataset/ransomware/ransomware_rwguard_features.csv", header=None)
ransomware_train_x = df_benign_train.iloc[:, :-1].values
ransomware_train_y = [1] * len(ransomware_train_x)

df_benign_test = pd.read_csv("./dataset/benign/benign_rwguard_features_3sec_test.csv", header=None)
benign_test_x = df_benign_test.iloc[:, :-1].values
benign_test_y = [0] * len(benign_test_x)

df_ransomware_test = pd.read_csv("./dataset/ransomware/ransomware_rwguard_features_test.csv", header=None)
ransomware_test_x = df_ransomware_test.iloc[:, :-1].values
ransomware_test_y = [1] * len(ransomware_test_x)


# you can play with the number of estimators and tree max depth parameter to build and then explain different models. select n_jobs at lest 2 less than the number of you CPU cores
clf = RandomForestClassifier(n_estimators=100, verbose=1, max_depth=100, n_jobs=10) #https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html


train_x = np.concatenate((benign_train_x, ransomware_train_x), axis=0)
train_y = np.concatenate((benign_train_y, ransomware_train_y), axis=0)

test_x = np.concatenate((benign_test_x, ransomware_test_x), axis=0)
test_y = np.concatenate((benign_test_y, ransomware_test_y), axis=0)

rng = np.random.default_rng()
indices = rng.permutation(len(train_x))

train_x = train_x[indices]
train_y = train_y[indices]

clf.fit(train_x, train_y)

dump(clf, 'rwguard_model.joblib') # using joblib to save the model for later load and use, there are other ways to store/load models


#evaluate the model here on the test data and print performance metrics. See sklearn documentation https://scikit-learn.org/stable/api/sklearn.metrics.html
y_pred = clf.predict(test_x)

print("Accuracy:", metrics.accuracy_score(test_y, y_pred)) # pay attention to the ordering
