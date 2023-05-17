from distutils.command.clean import clean
import pandas as pd
import numpy as np
import seaborn as sns
import csv
from scipy import stats
import matplotlib.pyplot as plt

data = pd.read_csv('kopi.csv')

#data.isna().sum()
def dataset():
	ganti = "?"
	dataset = data.copy()
	dataset = dataset.fillna(ganti)
	dataset.to_csv('dataset.csv')
	return dataset