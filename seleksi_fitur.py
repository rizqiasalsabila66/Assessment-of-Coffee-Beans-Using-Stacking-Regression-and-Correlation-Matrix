from distutils.command.clean import clean
import pandas as pd
import numpy as np
import seaborn as sns
import csv
from scipy import stats
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


raw_data = pd.read_csv('clean_data.csv')

X = raw_data.drop(['Total.Cup.Points', 'Unnamed: 0', 'Unnamed: 0.1'], axis=1)
X.shape
y = raw_data['Total.Cup.Points']
y.shape

def seleksi_fitur():
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)
	X_train.shape, y_train.shape, X_test.shape, y_test.shape
	# Build a Dataframe with Correlation between Features
	corr_matrix = X_train.corr()
	# Take absolute values of correlated coefficients
	corr_matrix = corr_matrix.abs().unstack()
	corr_matrix = corr_matrix.sort_values(ascending=False)
	# Take only features with correlation above threshold of 0.2
	corr_matrix = corr_matrix[corr_matrix >= 0.2]
	corr_matrix = corr_matrix[corr_matrix < 1]
	corr_matrix = pd.DataFrame(corr_matrix).reset_index()
	corr_matrix.columns = ['feature1', 'feature2', 'Correlation']
	corr_matrix.head()
	# Get groups of features that are correlated amongs themselves
	grouped_features = []
	correlated_groups = []
	for feature in corr_matrix.feature1.unique():
		if feature not in grouped_features:
			correlated_block = corr_matrix[corr_matrix.feature1 == feature]
			grouped_features = grouped_features + list(correlated_block.feature2.unique()) + [feature]
			correlated_groups.append(correlated_block)
	group_1 = correlated_groups[1]
	fitur = list(group_1.feature2.unique())+ list(group_1.feature1.unique()) + ['Total.Cup.Points']
	seleksi_fitur_data = raw_data.copy()
	seleksi_fitur_data = raw_data[fitur]
	seleksi_fitur_data.to_csv('seleksi_fitur_data.csv')
	return seleksi_fitur_data
