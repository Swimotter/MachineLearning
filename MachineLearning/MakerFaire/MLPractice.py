#Following tutorial: https://towardsdatascience.com/how-to-build-your-first-machine-learning-model-in-python-e70fd1907cdd


import sys
import scipy as sp
import numpy as np
import matplotlib
import pandas as pd
import sklearn as sk

#assign training and test data
from sklearn.model_selection import train_test_split
#training models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
#data analysis
from sklearn.metrics import mean_squared_error, r2_score
#data visualization
from matplotlib import pyplot as plt

#get csv file, set x to all column except last, set y to last column
df = pd.read_csv('https://github.com/dataprofessor/data/raw/master/delaney_solubility_with_descriptors.csv')
x = df.drop(['logS'], axis = 1)
y = df.iloc[:,-1]

#assign 80% of the data to training and 20% to testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

#linear regression model
lr = LinearRegression()
lr.fit(x_train, y_train)
y_lr_train_pred = lr.predict(x_train)
y_lr_test_pred = lr.predict(x_test)

lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)
lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

lr_results = pd.DataFrame(['Linear regression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ['Method','Training MSE','Training R2','Test MSE','Test R2']

#random forest model
rf = RandomForestRegressor(max_depth = 2, random_state = 2)
rf.fit(x_train, y_train)
y_rf_train_pred = rf.predict(x_train)
y_rf_test_pred = rf.predict(x_test)

rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)
rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)

rf_results = pd.DataFrame(['Random forest', rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
rf_results.columns = ['Method','Training MSE','Training R2','Test MSE','Test R2']

#extra tree model
et = ExtraTreesRegressor(max_depth = 2, random_state = 2)
et.fit(x_train, y_train)
y_et_train_pred = et.predict(x_train)
y_et_test_pred = et.predict(x_test)

et_train_mse = mean_squared_error(y_train, y_et_train_pred)
et_train_r2 = r2_score(y_train, y_et_train_pred)
et_test_mse = mean_squared_error(y_test, y_et_test_pred)
et_test_r2 = r2_score(y_test, y_et_test_pred)

et_results = pd.DataFrame(['Extra tree', et_train_mse, et_train_r2, et_test_mse, et_test_r2]).transpose()
et_results.columns = ['Method','Training MSE','Training R2','Test MSE','Test R2']

#gradient boost model
gb = GradientBoostingRegressor(max_depth = 2, random_state = 2)
gb.fit(x_train, y_train)
y_gb_train_pred = gb.predict(x_train)
y_gb_test_pred = gb.predict(x_test)

gb_train_mse = mean_squared_error(y_train, y_gb_train_pred)
gb_train_r2 = r2_score(y_train, y_gb_train_pred)
gb_test_mse = mean_squared_error(y_test, y_gb_test_pred)
gb_test_r2 = r2_score(y_test, y_gb_test_pred)

gb_results = pd.DataFrame(['Gradient boost', gb_train_mse, gb_train_r2, gb_test_mse, gb_test_r2]).transpose()
gb_results.columns = ['Method','Training MSE','Training R2','Test MSE','Test R2']

#k-neighbors model
kn = KNeighborsRegressor(n_neighbors = 2)
kn.fit(x_train, y_train)
y_kn_train_pred = kn.predict(x_train)
y_kn_test_pred = kn.predict(x_test)

kn_train_mse = mean_squared_error(y_train, y_kn_train_pred)
kn_train_r2 = r2_score(y_train, y_kn_train_pred)
kn_test_mse = mean_squared_error(y_test, y_kn_test_pred)
kn_test_r2 = r2_score(y_test, y_kn_test_pred)

kn_results = pd.DataFrame(['K-neighbors', kn_train_mse, kn_train_r2, kn_test_mse, kn_test_r2]).transpose()
kn_results.columns = ['Method','Training MSE','Training R2','Test MSE','Test R2']

#machine learning model
ml = RandomForestRegressor(max_depth = 2, random_state = 2)
ml.fit(x_train, y_train)
y_ml_train_pred = ml.predict(x_train)
y_ml_test_pred = ml.predict(x_test)

ml_train_mse = mean_squared_error(y_train, y_ml_train_pred)
ml_train_r2 = r2_score(y_train, y_ml_train_pred)
ml_test_mse = mean_squared_error(y_test, y_ml_test_pred)
ml_test_r2 = r2_score(y_test, y_ml_test_pred)

ml_results = pd.DataFrame(['Machine Learning', ml_train_mse, ml_train_r2, ml_test_mse, ml_test_r2]).transpose()
ml_results.columns = ['Method','Training MSE','Training R2','Test MSE','Test R2']

#combine dataframes
print(pd.concat([lr_results, rf_results, et_results, gb_results, kn_results, ml_results]))

#visualize data
plt.figure(figsize = (5,5))
plt.scatter(x = y_train, y = y_kn_train_pred, c = "#7CAE00", alpha=0.3)
z = np.polyfit(y_train, y_kn_train_pred, 1)
p = np.poly1d(z)
plt.plot(y_train,p(y_train),"#F8766D")
plt.ylabel('Predicted LogS')
plt.xlabel('Experimental LogS')
plt.title('Gradient Boost')
plt.show()