import pandas as pd
import numpy as np
import seaborn as sns
import csv
from scipy import stats

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from mlxtend.regressor import StackingRegressor
from mlxtend.data import boston_housing_data

import matplotlib.pyplot as plt
import warnings

df = pd.read_csv('seleksi_fitur_data.csv')
#df.head()
X = df.drop(['Total.Cup.Points'], axis=1)
X.shape
y = df['Total.Cup.Points']
y.shape 
  
# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.10, random_state= 123)

#scale data

scaler= StandardScaler()

X_train_scaled= scaler.fit_transform(X_train)
X_test_scaled= scaler.transform(X_test)
def akurasi():
	def train_test_evaluate(model_name, model, X_train, y_train, X_test, y_test):
		model.fit(X_train, y_train)
		y_pred= model.predict(X_test)
		mse = mean_squared_error(y_test, y_pred)
		rmse = np.sqrt(mean_squared_error(y_test, y_pred))
		r2 = model.score(X_test, y_test.values)
		y_test_mean= y_test.mean()
		result_df = pd.DataFrame(
			data=[[model_name, mse,rmse,  r2]], 
			columns=["Model", 'MSE','RMSE', "R2 Score"])
		return result_df

	knn= KNeighborsRegressor(leaf_size= 5, n_neighbors = 10)
	svr = SVR(kernel='linear', C=0.01)
	rf= RandomForestRegressor(n_estimators = 100, max_depth = 100)
	stregr_model = StackingRegressor(regressors=[knn, svr], meta_regressor= rf)
	result_df= train_test_evaluate("Stacking Regressor", stregr_model, X_train_scaled, y_train, X_test_scaled, y_test)
	result_df

	rf_model= RandomForestRegressor(n_estimators = 100, max_depth = 100)

	result_df_2= train_test_evaluate("Random Forest", rf_model, X_train_scaled, y_train, X_test_scaled, y_test)
	result_df = result_df.append(result_df_2, ignore_index=True)
	result_df= result_df.sort_values("R2 Score", ignore_index= True)
	result_df

	knn_model= KNeighborsRegressor(leaf_size= 5, n_neighbors = 10)

	result_df_2= train_test_evaluate("KNN", knn_model, X_train_scaled, y_train, X_test_scaled, y_test)
	result_df = result_df.append(result_df_2, ignore_index=True)
	result_df= result_df.sort_values("R2 Score", ignore_index= True)
	result_df

	svr_model= SVR(kernel='linear', C=0.01)

	result_df_2 = train_test_evaluate("SVR", svr_model, X_train_scaled, y_train, X_test_scaled, y_test)
	result_df = result_df.append(result_df_2, ignore_index=True)
	result_df= result_df.sort_values("R2 Score", ignore_index= True)
	result_df
	result_df.to_csv('akurasi.csv')
	return result_df
