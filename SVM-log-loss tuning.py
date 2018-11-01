# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn import preprocessing
from sklearn import svm
import time

# Loads the files
test_data_logloss = pd.read_csv('./log-loss/test_data.csv', header=None)
train_data_logloss = pd.read_csv('./log-loss/train_data.csv', header=None)
train_labels_logloss = pd.read_csv('./log-loss/train_labels.csv', header=None)

# Parse loaded content
test_data_logloss = test_data_logloss.values
train_data_logloss = train_data_logloss.values
train_labels_logloss = train_labels_logloss.values

class_names = ['Pop_Rock', 'Electronic', 'Rap', 'Jazz', 'Latin', 'RnB', 'International', 'Country', 'Reggae', 'Blues']

# Scales the data before feeding them into ML algorithms
scaler = preprocessing.StandardScaler().fit(train_data_logloss)
train_data_scaled = scaler.transform(train_data_logloss)
test_data_scaled = scaler.transform(test_data_logloss)
print(f'The mean of the train data is: {np.mean(train_data_scaled)}, the variance is {np.std(train_data_scaled)}')
print(f'The mean of the test data is: {np.mean(test_data_scaled)}, the variance is {np.std(test_data_scaled)}')

## Calculate the time needed for the following cell
## Determine the best parameters for the Support Vector Algorithm
parameters = {'kernel':('linear', 'poly', 'rbf', 'sigmoid'), 'C':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'gamma':('scale', 'auto')}
svc = svm.SVC(probability=True, verbose=True)
clf = GridSearchCV(svc, parameters, cv=5, return_train_score=True, n_jobs=-1)
clf.fit(train_data_scaled, np.ravel(train_labels_logloss))

print('The best estimator is:')
print(clf.best_estimator_)

