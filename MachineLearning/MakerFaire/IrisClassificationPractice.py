#Followed guide: https://machinelearningmastery.com/machine-learning-in-python-step-by-step/


import sys
import scipy as sp
import numpy as np
import matplotlib
import pandas as pd
import sklearn as sk

#libraries
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#load data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ["sepal-length", "sepal-width", 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names = names)

# shape
print(dataset.shape)
# head
print(dataset.head(20))
# descriptions
print(dataset.describe())
# class distribution
print(dataset.groupby('class').size())

#data visualization
dataset.plot(kind = "box", subplots = True, layout = (2,2), sharex = False, sharey = False)
dataset.hist()
scatter_matrix(dataset)
plt.show()

x = dataset.drop(['class'], axis = 1)
y = dataset[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 2)

#Algorithms
models = []
models.append('Logistic Regression', LogisticRegression(solver = 'liblinear', multi_class = 'ovr'))
models.append('Linear Discriminant', LinearDiscriminantAnalysis())
models.append('K Neighbors', KNeighborsClassifier())
models.append('Decision Tree', DecisionTreeClassifier())
models.append('Gaussian', GaussianNB())
models.append('SVM', SVC(gamma = 'auto'))

results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits = 10, random_state = 2, shuffle = True)
    cv_results = cross_val_score(model, x_train, y_train, cv = kfold, scoring = 'accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
