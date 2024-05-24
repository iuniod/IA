# remove first 10 attributes
import pandas as pd



def remove_correlated_attributes(file_name, attributes_to_remove):
	file = pd.read_csv(file_name)
	correlated_file = file.drop(columns=attributes_to_remove)
	return correlated_file

if __name__ == '__main__':
	# Salary Prediction dataset
	file_name = './tema2_SalaryPrediction/preprocessed_outliers_SalaryPrediction_train.csv'
	file_sufix = 'SalaryPrediction_train.csv'
	print(f'Preprocessing {file_sufix} dataset')
	attributes_to_remove = ['prod', 'gtype']
	correlated_file = remove_correlated_attributes(file_name, attributes_to_remove)
	correlated_file.to_csv(f'./tema2_SalaryPrediction/preprocessed_correlated_{file_sufix}', index=False)
	

	file_name = './tema2_SalaryPrediction/preprocessed_outliers_SalaryPrediction_test.csv'
	file_sufix = 'SalaryPrediction_test.csv'
	print(f'Preprocessing {file_sufix} dataset')
	attributes_to_remove = ['prod', 'gtype']
	correlated_file = remove_correlated_attributes(file_name, attributes_to_remove)
	correlated_file.to_csv(f'./tema2_SalaryPrediction/preprocessed_correlated_{file_sufix}', index=False)


	# Stroke Prediction dataset
	file_name = './tema2_AVC/preprocessed_outliers_AVC_train.csv'
	file_sufix = 'AVC_train.csv'
	print(f'Preprocessing {file_sufix} dataset')
	attributes_to_remove = ['analysis_results', 'chaotic_sleep']
	correlated_file = remove_correlated_attributes(file_name, attributes_to_remove)
	correlated_file.to_csv(f'./tema2_AVC/preprocessed_correlated_{file_sufix}', index=False)


	file_name = './tema2_AVC/preprocessed_outliers_AVC_test.csv'
	file_sufix = 'AVC_test.csv'
	print(f'Preprocessing {file_sufix} dataset')
	attributes_to_remove = ['analysis_results', 'chaotic_sleep']
	correlated_file = remove_correlated_attributes(file_name, attributes_to_remove)
	correlated_file.to_csv(f'./tema2_AVC/preprocessed_correlated_{file_sufix}', index=False)