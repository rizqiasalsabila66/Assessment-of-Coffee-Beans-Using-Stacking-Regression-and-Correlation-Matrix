import pandas as pd
import numpy as np
import seaborn as sns
import csv
from scipy import stats

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from mlxtend.regressor import StackingRegressor
from mlxtend.data import boston_housing_data

import matplotlib.pyplot as plt
import warnings
import pickle


df = pd.read_csv("seleksi_fitur_data.csv")
X = df[['Flavor', 'Balance', 'Acidity', 'Cupper.Points', 'Aroma','Body', 'Uniformity', 'Clean.Cup', 'Sweetness', 'Category.Two.Defects', 'Aftertaste']]
Y = df['Total.Cup.Points']

# train test split

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size= 0.1, random_state= 123)
#scale data

scaler= StandardScaler()
X_train= scaler.fit_transform(X_train)
X_test= scaler.transform(X_test)

knn= KNeighborsRegressor(leaf_size= 5, n_neighbors = 10)
svr = SVR(kernel='linear', C=0.01)
rf= RandomForestRegressor(n_estimators = 100, max_depth = 100)
stregr = StackingRegressor(regressors=[knn, svr], meta_regressor= rf)

stregr.fit(X_train, y_train)

pickle.dump(stregr, open("model.pkl", "wb"))

#print("Mean Squared Error: %.6f" % mean_squared_error(y_test, y_pred))
#print("RMSE : ", np.sqrt(mean_squared_error(y_test, y_pred)))
#print('R2 : %.6f' % stregr.score(X_test, y_test.values))
