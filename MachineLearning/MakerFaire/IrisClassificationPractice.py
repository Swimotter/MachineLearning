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

x = dataset.drop(['class'], axis = 1)
y = dataset.iloc[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 2)

#algorithms
models = []
models.append(['LR', LogisticRegression(solver = 'liblinear', multi_class = 'ovr')])
models.append(['LDA', LinearDiscriminantAnalysis()])
models.append(['KNN', KNeighborsClassifier()])
models.append(['CART', DecisionTreeClassifier()])
models.append(['NB', GaussianNB()])
models.append(['SVM', SVC(gamma = 'auto')])

results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits = 10, random_state = 2, shuffle = True)
    cv_results = cross_val_score(model, x_train, y_train, cv = kfold, scoring = 'accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

#predictions
model = SVC(gamma = 'auto', probability = True)
model.fit(x_train, y_train)
predictions = model.predict(x_test)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

xNew = [[6.3, 2.8, 4.7, 1.3]]
yNew = model.predict_proba(xNew)

print('x = %s, Prediction: Iris-setosa(%s%%) Iris-versicolor(%s%%) Iris-virginica(%s%%)' % (xNew[0], yNew[0][0] * 100, yNew[0][1], yNew[0][2] * 100))

#data visualization
dataset.plot(kind = "box", subplots = True, layout = (2,2), sharex = False, sharey = False)
dataset.hist()
scatter_matrix(dataset)

#algorithms visualization
plt.figure(figsize = (5,5))
plt.boxplot(results, labels = names)
plt.title('Algorithm Comparison')


plt.show()