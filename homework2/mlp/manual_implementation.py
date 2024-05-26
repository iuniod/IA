from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import List
import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler


class MLP(nn.Module):
	def __init__(self, input_size, hidden_sizes, output_size):
		super(MLP, self).__init__()
		layers = []
		in_size = input_size
		for hidden_size in hidden_sizes:
			layers.append(nn.Linear(in_size, hidden_size))
			layers.append(nn.Sigmoid())
			in_size = hidden_size
		layers.append(nn.Linear(in_size, output_size))
		layers.append(nn.LogSoftmax(dim=1))
		self.network = nn.Sequential(*layers)

	def forward(self, x):
		return self.network(x)

	def fit(self, X_train, y_train, num_epochs=100, learning_rate=0.01):
		X_train = torch.tensor(X_train, dtype=torch.float32)
		y_train = torch.tensor(y_train, dtype=torch.long)

		criterion = nn.NLLLoss()
		optimizer = optim.SGD(self.parameters(), lr=learning_rate)

		self.train()
		for epoch in range(num_epochs):
			optimizer.zero_grad()
			outputs = self(X_train)
			loss = criterion(outputs, y_train)
			loss.backward()
			optimizer.step()

	def predict(self, X):
		X = torch.tensor(X, dtype=torch.float32)
		self.eval()
		with torch.no_grad():
			outputs = self(X)
			_, preds = torch.max(outputs, 1)
		return preds.numpy()

def nll(Y, T):
    """Compute the negative log-likelihood (NLL) for binary classification."""
    N = T.shape[0]  # Number of samples
    return -np.sum(T * np.log(Y) + (1 - T) * np.log(1 - Y)) / N  # Calculate NLL


def plot_learning_curves(train_acc, test_acc, train_loss, test_loss, title, dataset):
	"""Plot learning curves for training and test accuracy and loss."""
	epochs = range(1, len(train_acc) + 1)  # Generate a range of epochs
	plt.figure(figsize=(14, 6))  # Create a figure for the plots

	plt.subplot(1, 2, 1)  # Create a subplot for accuracy
	plt.plot(epochs, train_acc, 'bo-', label='Train Accuracy')  # Plot training accuracy
	plt.plot(epochs, test_acc, 'ro-', label='Test Accuracy')  # Plot test accuracy
	plt.title(f'{title} - Accuracy')  # Set the title for the accuracy plot
	plt.xlabel('Epochs')  # Label the x-axis
	plt.ylabel('Accuracy')  # Label the y-axis
	plt.legend()  # Add a legend

	plt.subplot(1, 2, 2)  # Create a subplot for loss
	plt.plot(epochs, train_loss, 'bo-', label='Train Loss')  # Plot training loss
	plt.plot(epochs, test_loss, 'ro-', label='Test Loss')  # Plot test loss
	plt.title(f'{title} - Loss')  # Set the title for the loss plot
	plt.xlabel('Epochs')  # Label the x-axis
	plt.ylabel('Loss')  # Label the y-axis
	plt.legend()  # Add a legend

	plt.tight_layout()  # Adjust subplots to fit into the figure area

	# Save the learning curves
	plt.savefig(f'./tema2_{dataset}/MLP/learning_curves_manual.png')

def solver(dataset):
	train_file = f'./tema2_{dataset}/preprocessed_standardized_{dataset}_train_converted.csv'
	test_file = f'./tema2_{dataset}/preprocessed_standardized_{dataset}_test_converted.csv'
	
	# Load the datasets
	train = pd.read_csv(train_file)
	test = pd.read_csv(test_file)

	# Get the target column
	categorical_attributes = []
	with open(f'./tema2_{dataset}/categorical_attributes.txt') as f:
		categorical_attributes = f.read().splitlines()
	target = categorical_attributes[-1]

	# Separate features and target
	X_train = train.drop(target, axis=1)
	y_train = train[target].values

	X_test = test.drop(target, axis=1)
	y_test = test[target].values

	# # Standardize features
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)

	input_size = X_train.shape[1]
	hidden_sizes = [256, 128, 64]
	output_size = len(np.unique(y_train))  # Number of classes

	# Create and train the MLP model
	print(f"Training MLP {dataset} model...")
	model = MLP(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size)
	model.fit(X_train, y_train)

	# Make predictions
	print(f"Making predictions for {dataset} dataset...")
	y_train_pred = model.predict(X_train)
	y_test_pred = model.predict(X_test)

	# Evaluate the model
	print(f"Evaluating the model for {dataset} dataset...")
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

	# Check if the directory exists
	if not os.path.exists(f'./tema2_{dataset}/MLP'):
		os.makedirs(f'./tema2_{dataset}/MLP')

	# Save the confusion matrix
	# plt.savefig(f'./tema2_{dataset}/MLP/confusion_matrix_manual.png')

	# # Save the metrics
	# with open(f'./tema2_{dataset}/MLP/metrics_manual.txt', 'w') as f:
	#     f.write(f"Training accuracy: {train_accuracy}\n")
	#     f.write(f"Training precision: {train_precision}\n")
	#     f.write(f"Training recall: {train_recall}\n")
	#     f.write(f"Training f1: {train_f1}\n")
	#     f.write(f"Test accuracy: {test_accuracy}\n")
	#     f.write(f"Test precision: {test_precision}\n")
	#     f.write(f"Test recall: {test_recall}\n")
	#     f.write(f"Test f1: {test_f1}\n")

	# Plot learning curves
	num_epochs = 10
	# generate a range of epochs
	epochs = range(1, num_epochs + 1)
	train_acc = []
	test_acc = []
	train_loss = []
	test_loss = []

	for epoch in epochs:
		model = MLP(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size)
		model.fit(X_train, y_train)
		y_train_pred = model.predict(X_train)
		y_test_pred = model.predict(X_test)
		train_acc.append(accuracy_score(y_train, y_train_pred))
		test_acc.append(accuracy_score(y_test, y_test_pred))
		train_loss.append(nll(y_train_pred, y_train))
		test_loss.append(nll(y_test_pred, y_test))
	
	plot_learning_curves(train_acc, test_acc, train_loss, test_loss, f'{dataset} - MLP', dataset)


if __name__ == '__main__':
	solver('SalaryPrediction')
	solver('AVC')
