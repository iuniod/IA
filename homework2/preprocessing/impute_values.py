import numpy as np
import pandas as pd
import os
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from scipy import stats


def missing_values(file_name):
	file = pd.read_csv(file_name)

	# convert ? to NaN
	file = file.replace('?', np.nan)

	# Separate numeric and categorical columns
	with open(os.path.join(os.path.dirname(file_name), 'numeric_attributes.txt'), 'r') as f:
		numerical_cols = f.read().splitlines()

    # Generate box plots for numerical attributes
	with open(os.path.join(os.path.dirname(file_name), 'categorical_attributes.txt'), 'r') as f:
		categorical_cols = f.read().splitlines()

	# Impute missing values in numeric columns
	imputer = IterativeImputer(max_iter=10, random_state=0)
	imputed_numeric = imputer.fit_transform(file[numerical_cols])

	# Impute missing values in categorical columns: missing values can contain ? or NaN or nothing
	# Use the most frequent value for each column
	imputed_categorical = file[categorical_cols].apply(lambda x: x.fillna(x.value_counts().index[0]))

	# Concatenate the imputed numeric and categorical columns
	imputed_file = pd.concat([pd.DataFrame(imputed_numeric, columns=numerical_cols), imputed_categorical], axis=1)

	return imputed_file
	
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

		# Replace the outliers with the median value
		file.loc[outliers, col] = file[col].median()

	return file

if __name__ == '__main__':
	# Salary Prediction dataset
	file_name = './tema2_SalaryPrediction/SalaryPrediction_train.csv'
	file_sufix = 'SalaryPrediction_train.csv'
	print(f'Preprocessing {file_sufix} dataset')
	imputed_file = missing_values(file_name)
	imputed_file.to_csv(f'./tema2_SalaryPrediction/preprocessed_missing_{file_sufix}', index=False)

	if imputed_file.isnull().values.any():
		print('The file still contains missing values')
	else:
		print('All missing values have been imputed')

	file_name = './tema2_SalaryPrediction/preprocessed_missing_SalaryPrediction_train.csv'
	imputed_file = impute_outliers(file_name)
	imputed_file.to_csv(f'./tema2_SalaryPrediction/preprocessed_outliers_{file_sufix}', index=False)


	file_name = './tema2_SalaryPrediction/SalaryPrediction_test.csv'
	file_sufix = 'SalaryPrediction_test.csv'
	print(f'Preprocessing {file_sufix} dataset')
	imputed_file = missing_values(file_name)
	imputed_file.to_csv(f'./tema2_SalaryPrediction/preprocessed_missing_{file_sufix}', index=False)

	if imputed_file.isnull().values.any():
		print('The file still contains missing values')
	else:
		print('All missing values have been imputed')

	file_name = './tema2_SalaryPrediction/preprocessed_missing_SalaryPrediction_test.csv'
	imputed_file = impute_outliers(file_name)
	imputed_file.to_csv(f'./tema2_SalaryPrediction/preprocessed_outliers_{file_sufix}', index=False)


	# Stroke Prediction dataset
	file_name = './tema2_AVC/AVC_train.csv'
	file_sufix = 'AVC_train.csv'
	print(f'Preprocessing {file_sufix} dataset')
	imputed_file = missing_values(file_name)
	imputed_file.to_csv(f'./tema2_AVC/preprocessed_missing_{file_sufix}', index=False)

	if imputed_file.isnull().values.any():
		print('The file still contains missing values')
	else:
		print('All missing values have been imputed')

	file_name = './tema2_AVC/preprocessed_missing_AVC_train.csv'
	imputed_file = impute_outliers(file_name)
	imputed_file.to_csv(f'./tema2_AVC/preprocessed_outliers_{file_sufix}', index=False)

	
	file_name = './tema2_AVC/AVC_test.csv'
	file_sufix = 'AVC_test.csv'
	print(f'Preprocessing {file_sufix} dataset')
	imputed_file = missing_values(file_name)
	imputed_file.to_csv(f'./tema2_AVC/preprocessed_missing_{file_sufix}', index=False)

	if imputed_file.isnull().values.any():
		print('The file still contains missing values')
	else:
		print('All missing values have been imputed')

	file_name = './tema2_AVC/preprocessed_missing_AVC_test.csv'
	imputed_file = impute_outliers(file_name)
	imputed_file.to_csv(f'./tema2_AVC/preprocessed_outliers_{file_sufix}', index=False)