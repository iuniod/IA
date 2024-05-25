import numpy as np
import pandas as pd
import os
from sklearn.impute import SimpleImputer
from scipy import stats


def missing_values(file_name, imputers=None):
	file = pd.read_csv(file_name)

	# convert ? to NaN
	file = file.replace('?', np.nan)

	# Separate numeric and categorical columns
	with open(os.path.join(os.path.dirname(file_name), 'numeric_attributes.txt'), 'r') as f:
		numerical_cols = f.read().splitlines()

    # Generate box plots for numerical attributes
	with open(os.path.join(os.path.dirname(file_name), 'categorical_attributes.txt'), 'r') as f:
		categorical_cols = f.read().splitlines()

	numeric_imputer = SimpleImputer(strategy='mean')
	categorical_imputer = SimpleImputer(strategy='most_frequent')
	if imputers is not None:
		numeric_imputer = imputers['numeric']
		categorical_imputer = imputers['categorical']

	# Impute missing values in numeric columns
	df_numeric_imputed = pd.DataFrame(numeric_imputer.fit_transform(file[numerical_cols]), columns=numerical_cols)

	# Impute missing values in categorical columns
	df_categorical_imputed = pd.DataFrame(categorical_imputer.fit_transform(file[categorical_cols]), columns=categorical_cols)

	# Store imputers for future use
	imputers = {'numeric': numeric_imputer, 'categorical': categorical_imputer}

	# Concatenate the imputed numeric and categorical columns
	imputed_file = pd.concat([df_numeric_imputed, df_categorical_imputed], axis=1)

	return imputed_file, imputers
	
def impute_outliers(file_name):
	file = pd.read_csv(file_name)
	
	# Separate numeric columns
	with open(os.path.join(os.path.dirname(file_name), 'numeric_attributes.txt'), 'r') as f:
		numerical_cols = f.read().splitlines()

	# Calculate Q1 and Q3 for each numerical column
	for col in numerical_cols:
		Q1 = file[col].quantile(0.25)
		Q3 = file[col].quantile(0.75)
		IQR = Q3 - Q1

		# Get the indices of the outliers
		threshold = 1.5
		outliers = file[(file[col] < Q1 - threshold * IQR) | (file[col] > Q3 + threshold * IQR)].index

		# Replace the outliers with NaN
		file.loc[outliers, col] = np.nan
	
	# Impute the outliers
	imputer = SimpleImputer(strategy='mean')
	imputed_file = pd.DataFrame(imputer.fit_transform(file[numerical_cols]), columns=numerical_cols)

	# Concatenate the imputed numeric and categorical columns
	file[numerical_cols] = imputed_file

	return file

if __name__ == '__main__':
	# Salary Prediction dataset
	# Train dataset
	file_name = './tema2_SalaryPrediction/SalaryPrediction_train.csv'
	file_sufix = 'SalaryPrediction_train.csv'
	print(f'Preprocessing {file_sufix} dataset')
	imputed_file, imputers = missing_values(file_name)
	imputed_file.to_csv(f'./tema2_SalaryPrediction/preprocessed_missing_{file_sufix}', index=False)

	if imputed_file.isnull().values.any():
		print('The file still contains missing values')
	else:
		print('All missing values have been imputed')

	file_name = './tema2_SalaryPrediction/preprocessed_missing_SalaryPrediction_train.csv'
	imputed_file = impute_outliers(file_name)
	imputed_file.to_csv(f'./tema2_SalaryPrediction/preprocessed_outliers_{file_sufix}', index=False)


	# Test dataset
	file_name = './tema2_SalaryPrediction/SalaryPrediction_test.csv'
	file_sufix = 'SalaryPrediction_test.csv'
	print(f'Preprocessing {file_sufix} dataset')
	imputed_file, _ = missing_values(file_name, imputers)
	imputed_file.to_csv(f'./tema2_SalaryPrediction/preprocessed_missing_{file_sufix}', index=False)

	if imputed_file.isnull().values.any():
		print('The file still contains missing values')
	else:
		print('All missing values have been imputed')

	file_name = './tema2_SalaryPrediction/preprocessed_missing_SalaryPrediction_test.csv'
	imputed_file = impute_outliers(file_name)
	imputed_file.to_csv(f'./tema2_SalaryPrediction/preprocessed_outliers_{file_sufix}', index=False)


	# Stroke Prediction dataset
	# Train dataset
	file_name = './tema2_AVC/AVC_train.csv'
	file_sufix = 'AVC_train.csv'
	print(f'Preprocessing {file_sufix} dataset')
	imputed_file, imputers = missing_values(file_name)
	imputed_file.to_csv(f'./tema2_AVC/preprocessed_missing_{file_sufix}', index=False)

	if imputed_file.isnull().values.any():
		print('The file still contains missing values')
	else:
		print('All missing values have been imputed')

	file_name = './tema2_AVC/preprocessed_missing_AVC_train.csv'
	imputed_file = impute_outliers(file_name)
	imputed_file.to_csv(f'./tema2_AVC/preprocessed_outliers_{file_sufix}', index=False)

	
	# Test dataset
	file_name = './tema2_AVC/AVC_test.csv'
	file_sufix = 'AVC_test.csv'
	print(f'Preprocessing {file_sufix} dataset')
	imputed_file, _ = missing_values(file_name, imputers)
	imputed_file.to_csv(f'./tema2_AVC/preprocessed_missing_{file_sufix}', index=False)

	if imputed_file.isnull().values.any():
		print('The file still contains missing values')
	else:
		print('All missing values have been imputed')

	file_name = './tema2_AVC/preprocessed_missing_AVC_test.csv'
	imputed_file = impute_outliers(file_name)
	imputed_file.to_csv(f'./tema2_AVC/preprocessed_outliers_{file_sufix}', index=False)
