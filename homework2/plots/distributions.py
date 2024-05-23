import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
  

if __name__ == "__main__":
    # Salary Prediction datasets
	full_set = pd.read_csv('./tema2_SalaryPrediction/SalaryPrediction_full.csv')
	train_set = pd.read_csv('./tema2_SalaryPrediction/SalaryPrediction_train.csv')
	test_set = pd.read_csv('./tema2_SalaryPrediction/SalaryPrediction_test.csv')

	full_set_features = full_set.select_dtypes(include=['object', 'category']).columns
	train_set_features = train_set.select_dtypes(include=['object', 'category']).columns
	test_set_features = test_set.select_dtypes(include=['object', 'category']).columns

	for feature in full_set_features:
		plt.figure()
		# convert nan values to string
		full_set[feature] = full_set[feature].astype(str)
		train_set[feature] = train_set[feature].astype(str)
		test_set[feature] = test_set[feature].astype(str)

		plt.hist(full_set[feature], bins=50, alpha=0.5, label='Full Dataset')
		plt.hist(train_set[feature], bins=50, alpha=0.5, label='Train Dataset')
		plt.hist(test_set[feature], bins=50, alpha=0.5, label='Test Dataset')

		if len(full_set[feature].unique()) > 10:
			plt.xticks(rotation=45)
		
		plt.title(f'Distribution of Salary for {feature}')
		plt.legend()
		plt.savefig(f'plots/distribution_{feature}_Salary_Prediction.png', dpi=300)
		plt.close()
	
	# Stroke Prediction datasets
	full_set = pd.read_csv('./tema2_AVC/AVC_full.csv')
	train_set = pd.read_csv('./tema2_AVC/AVC_train.csv')
	test_set = pd.read_csv('./tema2_AVC/AVC_test.csv')

	full_set_features = full_set.select_dtypes(include=['object', 'category']).columns
	train_set_features = train_set.select_dtypes(include=['object', 'category']).columns
	test_set_features = test_set.select_dtypes(include=['object', 'category']).columns

	for feature in full_set_features:
		plt.figure()
		# convert nan values to string
		full_set[feature] = full_set[feature].astype(str)
		train_set[feature] = train_set[feature].astype(str)
		test_set[feature] = test_set[feature].astype(str)

		plt.hist(full_set[feature], bins=50, alpha=0.5, label='Full Dataset')
		plt.hist(train_set[feature], bins=50, alpha=0.5, label='Train Dataset')
		plt.hist(test_set[feature], bins=50, alpha=0.5, label='Test Dataset')

		if len(full_set[feature].unique()) > 10:
			plt.xticks(rotation=45)
		
		plt.title(f'Distribution of Stroke for {feature}')
		plt.legend()
		plt.savefig(f'plots/distribution_{feature}_Stroke_Prediction.png', dpi=300)
		plt.close()

