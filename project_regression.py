import numpy as np 
import scipy as sp 
import sklearn as sk
from sklearn import tree
from sklearn.linear_model import LogisticRegression

dataset = np.genfromtxt('NBA_train.csv', delimiter = ',')
test_dataset = np.genfromtxt('NBA_test.csv', delimiter = ',')

X_test = test_dataset[:,[4, 5, 6, 7, 8, 9, 10, 11]]
Y_test = test_dataset[:, 2]

X = dataset[:,[4, 5, 6, 7, 8, 9, 10, 11]]
Y = dataset[:, 2]

logreg = LogisticRegression()

logreg.fit(X, Y)
predictions =  logreg.predict(X_test)
print predictions

expected = Y_test

summary = sk.metrics.classification_report(expected, predictions)
print summary



