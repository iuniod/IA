import numpy as np
import os
import sys
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge as RidgeRegression
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix



if __name__ == "__main__":
	# get the type of dataset
	if len(sys.argv) < 2:
		print("Usage: python convert_categorical_attributes.py <dataset>")
		sys.exit(1)
	dataset = sys.argv[1]
	train_file = f'./tema2_{dataset}/preprocessed_correlated_{dataset}_train_standardized_one_hot.csv'
	test_file = f'./tema2_{dataset}/preprocessed_correlated_{dataset}_test_standardized_one_hot.csv'
	
	# Load the datasets
	train = pd.read_csv(train_file)
	test = pd.read_csv(test_file)

	# get the target column
	categorical_attributes = []
	with open(f'./tema2_{dataset}/categorical_attributes.txt') as f:
		categorical_attributes = f.read().splitlines()
	target = categorical_attributes[-1]

	X_train = train.drop(target, axis=1).values
	y_train = train[target].values

	X_test = test.drop(target, axis=1).values
	y_test = test[target].values

	# Create and train the linear regression model
	model =  RidgeRegression(alpha=1.75)
	model.fit(X_train, y_train)

	# Make predictions
	y_train_pred = model.predict(X_train)
	y_test_pred = model.predict(X_test)

	# convert the predictions to binary
	# get the maximum value of the predictions
	mean_train = np.mean(y_train_pred)
	y_train_pred = np.where(y_train_pred > mean_train, 1, 0)
	mean_test = np.mean(y_test_pred)
	y_test_pred = np.where(y_test_pred > mean_test, 1, 0)

	# Evaluate the model
	# Training metrics
	train_accuracy = accuracy_score(y_train, y_train_pred)
	train_precision = precision_score(y_train, y_train_pred, average='weighted')
	train_recall = recall_score(y_train, y_train_pred, average='weighted')
	train_f1 = f1_score(y_train, y_train_pred, average='weighted')

	# Test metrics
	test_accuracy = accuracy_score(y_test, y_test_pred)
	test_precision = precision_score(y_test, y_test_pred, average='weighted')
	test_recall = recall_score(y_test, y_test_pred, average='weighted')
	test_f1 = f1_score(y_test, y_test_pred, average='weighted')

	# Confusion matrix
	train_cm = confusion_matrix(y_train, y_train_pred)
	test_cm = confusion_matrix(y_test, y_test_pred)

	# Plot the confusion matrix : write positive and negative labels and actual and predicted labels
	plt.figure(figsize=(10, 7))
	plt.subplot(1, 2, 1)
	sns.heatmap(train_cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
	plt.title('Train confusion matrix')
	plt.xlabel('Predicted')
	plt.ylabel('Actual')
	plt.subplot(1, 2, 2)
	sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
	plt.title('Test confusion matrix')
	plt.xlabel('Predicted')
	plt.ylabel('Actual')
	plt.tight_layout()
	
	# Save the confusion matrix
	plt.savefig(f'./tema2_{dataset}/confusion_matrix.png')

	# Save the metrics
	with open(f'./tema2_{dataset}/metrics.txt', 'w') as f:
		f.write(f"Training accuracy: {train_accuracy}\n")
		f.write(f"Training precision: {train_precision}\n")
		f.write(f"Training recall: {train_recall}\n")
		f.write(f"Training f1: {train_f1}\n")
		f.write(f"Test accuracy: {test_accuracy}\n")
		f.write(f"Test precision: {test_precision}\n")
		f.write(f"Test recall: {test_recall}\n")
		f.write(f"Test f1: {test_f1}\n")
