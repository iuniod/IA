from sklearn import preprocessing
import numpy as np
import pandas as pd
import os

def standardize(train_file, test_file):
	train = pd.read_csv(train_file)
	test = pd.read_csv(test_file)

	# Extract the features and the target
	header = train.columns
	print(header)
	X_train = train.iloc[:, :-1].values
	y_train = train.iloc[:, -1].values

	X_test = test.iloc[:, :-1].values
	y_test = test.iloc[:, -1].values

	# Separates categorical and numerical attributes
	categorical_attributes = []
	numerical_attributes = []
	with open(os.path.join(os.path.dirname(train_file), 'numeric_attributes.txt'), 'r') as f:
		numerical_attributes = f.read().splitlines()

	for col in numerical_attributes:
		if col not in train.columns:
			numerical_attributes.remove(col)

	with open(os.path.join(os.path.dirname(train_file), 'categorical_attributes.txt'), 'r') as f:
		categorical_attributes = f.read().splitlines()

	# Standardize only the numerical attributes
	scaler = preprocessing.StandardScaler()
	X_train[:, [train.columns.get_loc(col) for col in numerical_attributes]] = scaler.fit_transform(X_train[:, [train.columns.get_loc(col) for col in numerical_attributes]])
	X_test[:, [test.columns.get_loc(col) for col in numerical_attributes]] = scaler.transform(X_test[:, [test.columns.get_loc(col) for col in numerical_attributes]])

	# Save the standardized datasets and add headers
	train_standardized = pd.DataFrame(X_train, columns=header[:-1])
	train_standardized[header[-1]] = y_train
	test_standardized = pd.DataFrame(X_test, columns=header[:-1])
	test_standardized[header[-1]] = y_test

	train_standardized.to_csv(train_file.replace('.csv', '_standardized.csv'), index=False)
	test_standardized.to_csv(test_file.replace('.csv', '_standardized.csv'), index=False)

if __name__ == '__main__':
	# Salary Prediction dataset
	train_file = './tema2_SalaryPrediction/preprocessed_correlated_SalaryPrediction_train.csv'
	test_file = './tema2_SalaryPrediction/preprocessed_correlated_SalaryPrediction_test.csv'
	standardize(train_file, test_file)

	# Stroke Prediction dataset
	train_file = './tema2_AVC/preprocessed_correlated_AVC_train.csv'
	test_file = './tema2_AVC/preprocessed_correlated_AVC_test.csv'
	standardize(train_file, test_file)
