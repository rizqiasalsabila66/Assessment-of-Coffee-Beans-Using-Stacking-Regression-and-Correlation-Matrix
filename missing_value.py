from distutils.command.clean import clean
import pandas as pd
import numpy as np
import seaborn as sns
import csv
from scipy import stats


data = pd.read_csv('kopi.csv')
numerics = ['int16', 'int32','int64', 'float16', 'float32', 'float64']
numerical_features = list(data.select_dtypes(include=numerics).columns)
data = data[numerical_features]
#data.isna().sum()

def missing_value():

		rata_quakers = data['Quakers'].mean()
		data['Quakers'] = data['Quakers'].fillna(rata_quakers)
		rata_altitude_low_meters  = data['altitude_low_meters'].mean()
		data['altitude_low_meters'] = data['altitude_low_meters'].fillna(rata_altitude_low_meters)
		rata_altitude_high_meters  = data['altitude_high_meters'].mean()
		data['altitude_high_meters'] = data['altitude_high_meters'].fillna(rata_altitude_high_meters)
		rata_altitude_mean_meters  = data['altitude_mean_meters'].mean()
		data['altitude_mean_meters'] = data['altitude_mean_meters'].fillna(rata_altitude_mean_meters)
		clean_data = data.copy()
		clean_data.to_csv('clean_data.csv')
		return clean_data