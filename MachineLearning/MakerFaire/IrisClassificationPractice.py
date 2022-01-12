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

#data visualization
scatter_matrix(dataset)
plt.show()