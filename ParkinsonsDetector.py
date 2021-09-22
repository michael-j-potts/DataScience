#Title: ParkinsonsDetector.py
#Description: This program evaluates Parkinsons data and predicts whether a patient may have PD
#Date: Sept 22nd, 2021
#Author: Michael Potts
#Version 1.0

#Further ideas:
#Consider refining with a SVM
#Consider evaluating other data sets with this classifier

import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#inspiration provided by: https://data-flair.training/blogs/python-machine-learning-project-detecting-parkinson-disease/

# import the data
# https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/
df = pd.read_csv("/path to your data set downloaded from UCI")
print("Importing Parkinsons' disease data...")
# display(df)

# Identify the features and labels
features = df.loc[:, df.columns != 'status'].values[:, 1:]
labels = df.loc[:, 'status'].values

# This set transforms features by scaling them to a given range (-1, 1)
scaler = MinMaxScaler((-1, 1))
x = scaler.fit_transform(features)
y = labels

# Now split the data set into an 80% 20% split randomly
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=101)
N = X_train.shape[0]
print("The number of patients being analyzed: ", N)

# Initialize Guassian and Bernoulli naive bayes
gnb = GaussianNB()
Y_predict_gnb = gnb.fit(X_train, Y_train).predict(X_test)

bnb = BernoulliNB()
Y_predict_bnb = bnb.fit(X_train, Y_train).predict(X_test)

rfc = RandomForestClassifier(n_estimators=20, max_depth=8, random_state=101)
Y_predict_rfc = rfc.fit(X_train, Y_train).predict(X_test)

print("Gaussian accuracy = ", np.round(100 * (1 - ((Y_test != Y_predict_gnb).sum() / N)), 3), "%")
print("Bernoulli accuracy = ", np.round(100 * (1 - ((Y_test != Y_predict_bnb).sum() / N)), 3), "%")
print("Random forest accuracy = ", np.round(100 * (1 - ((Y_test != Y_predict_rfc).sum() / N)), 3), "%")
