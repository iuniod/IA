import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def distribution_of_classes(filename_prefix):
	filename_train = filename_prefix + '_train.csv'
	filename_test = filename_prefix + '_test.csv'

	# get the distribution of classes
	train = pd.read_csv(filename_train)
	test = pd.read_csv(filename_test)

	last_column = train.columns[-1]

	train_class_distribution = train[last_column].value_counts()
	test_class_distribution = test[last_column].value_counts()

	# plot the distribution of classes
	plt.figure()
	plt.bar(train_class_distribution.index, train_class_distribution.values, alpha=0.5, label='train')
	plt.bar(test_class_distribution.index, test_class_distribution.values, alpha=0.5, label='test')
	plt.xticks(train_class_distribution.index)
	plt.legend()
	plt.title(f'Distribution of classes for {os.path.basename(filename_prefix)} dataset')
	plt.savefig(f'plots/distribution_{os.path.basename(filename_prefix)}.png', dpi=300)
	plt.close()

if __name__ == "__main__":
    # Salary Prediction datasets
	filename_prefix = './tema2_SalaryPrediction/SalaryPrediction'
	distribution_of_classes(filename_prefix)

	# Stroke Prediction datasets
	filename_prefix = './tema2_AVC/AVC'
	distribution_of_classes(filename_prefix)

