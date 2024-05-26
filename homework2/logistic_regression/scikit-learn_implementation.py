import numpy as np
import os
import sys
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import uniform
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def get_best_model(X_train, y_train, X_test, y_test):
	# Create the model
	model = LogisticRegression()

	# Create the hyperparameter space: regularization parameter, solver, max_iter
	hyperparameters = {
		'C': uniform(loc=0, scale=4),
		'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
		'max_iter': [200, 500, 1000]
	}

	# Create the RandomizedSearchCV object
	search = RandomizedSearchCV(model, hyperparameters)

	# Fit the model
	search.fit(X_train, y_train)

	# Get the best model
	best_model = search.best_estimator_

	# print the best hyperparameters
	print(f"Best hyperparameters: {search.best_params_}")

	# Make predictions
	y_train_pred = best_model.predict(X_train)
	y_test_pred = best_model.predict(X_test)

	return y_train_pred, y_test_pred

def solver(dataset):
	train_file = f'./tema2_{dataset}/preprocessed_standardized_{dataset}_train_converted.csv'
	test_file = f'./tema2_{dataset}/preprocessed_standardized_{dataset}_test_converted.csv'
	
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

	# Get the best parameters for the model
	print(f"Training Logistic Regression {dataset} model...")
	y_train_pred, y_test_pred = get_best_model(X_train, y_train, X_test, y_test)

	# Evaluate the model
	# Training metrics
	train_accuracy = accuracy_score(y_train, y_train_pred)
	train_precision = precision_score(y_train, y_train_pred, zero_division=1)
	train_recall = recall_score(y_train, y_train_pred, zero_division=1)
	train_f1 = f1_score(y_train, y_train_pred, zero_division=1)

	# Test metrics
	test_accuracy = accuracy_score(y_test, y_test_pred)
	test_precision = precision_score(y_test, y_test_pred, zero_division=1)
	test_recall = recall_score(y_test, y_test_pred, zero_division=1)
	test_f1 = f1_score(y_test, y_test_pred, zero_division=1)

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

	# check if the directory exists
	if not os.path.exists(f'./tema2_{dataset}/LogisticRegression'):
		os.makedirs(f'./tema2_{dataset}/LogisticRegression')

	# Save the confusion matrix
	plt.savefig(f'./tema2_{dataset}/LogisticRegression/confusion_matrix_scikit-learn.png')

	# Save the metrics
	with open(f'./tema2_{dataset}/LogisticRegression/metrics_scikit-learn.txt', 'w') as f:
		f.write(f"Training accuracy: {train_accuracy}\n")
		f.write(f"Training precision: {train_precision}\n")
		f.write(f"Training recall: {train_recall}\n")
		f.write(f"Training f1: {train_f1}\n")
		f.write(f"Test accuracy: {test_accuracy}\n")
		f.write(f"Test precision: {test_precision}\n")
		f.write(f"Test recall: {test_recall}\n")
		f.write(f"Test f1: {test_f1}\n")


if __name__ == "__main__":
	solver('SalaryPrediction')
	solver('AVC')
	