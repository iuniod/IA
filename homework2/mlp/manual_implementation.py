from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import List
import numpy as np
import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class Layer:
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
        
    def backward(self, x: np.ndarray, dy: np.ndarray) -> np.ndarray:
        raise NotImplementedError
        
    def update(self, *args, **kwargs):
        pass


class FeedForwardNetwork:
    def __init__(self, layers: List[Layer]):
        self.layers = layers
        
    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        self._inputs = []
        for layer in self.layers:
            if train:
                self._inputs.append(x)
            x = layer.forward(x)
        return x
    
    def backward(self, dy: np.ndarray) -> np.ndarray:
        for x, layer in reversed(zip(self._inputs, self.layers)):
            dy = layer.backward(x, dy)
        return dy
    
        del self._inputs
    
    def update(self, *args, **kwargs):
        for layer in self.layers:
            layer.update(*args, **kwargs)

class Linear(Layer):
    def __init__(self, insize: int, outsize: int) -> None:
        bound = np.sqrt(6. / insize)
        self.weight = np.random.uniform(-bound, bound, (insize, outsize))
        self.bias = np.zeros((outsize,))
        
        self.dweight = np.zeros_like(self.weight)
        self.dbias = np.zeros_like(self.bias)
   
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return np.dot(x, self.weight) + self.bias
    
    def backward(self, x: np.ndarray, dy: np.ndarray) -> np.ndarray:
        self.dweight = np.dot(self.x.T, dy)
        self.dbias = np.sum(dy, axis=0)
        return np.dot(dy, self.weight.T)
    
    def update(self, mode='SGD', lr=0.001, mu=0.9):
        if mode == 'SGD':
            self.weight -= lr * self.dweight
            self.bias -= lr * self.dbias
        else:
            raise ValueError('mode should be SGD, not ' + str(mode))


class ReLU(Layer):
    def __init__(self) -> None:
        pass
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def backward(self, x: np.ndarray, dy: np.ndarray) -> np.ndarray:
        return dy * (x > 0)


class CrossEntropy:
    def __init__(self):
        pass
    
    def softmax(self, x):
        exps = np.exp(x)
        return exps / np.sum(exps,axis = 1).reshape(-1,1)

    def forward(self, y: np.ndarray, t: np.ndarray) -> float:
        self.y = y
        self.t = t
        self.p = self.softmax(y)
        return -np.mean(np.log(self.p[np.arange(len(t)), t]))
    
    def backward(self, y: np.ndarray, t: np.ndarray) -> np.ndarray:
        self.p = self.softmax(y)
        self.p[np.arange(len(t)), t] -= 1
        return self.p / len(t)


class MLP:
	def __init__(self, insize: int = 122, outsize: int = 2) -> None:
		self.network = FeedForwardNetwork([
			Linear(insize, 100),
			ReLU(),
			Linear(100, outsize)
		])
		self.loss = CrossEntropy()
		
	def fit(self, X: np.ndarray, y: np.ndarray, lr: float = 0.001, mu: float = 0.9, epochs: int = 1000) -> None:
		for epoch in range(epochs):
			y_pred = self.network.forward(X)
			loss = self.loss.forward(y_pred, y)
			dy = self.loss.backward(y_pred, y)
			self.network.backward(dy)
			self.network.update(lr=lr, mu=mu)
	
	def predict(self, X: np.ndarray) -> np.ndarray:
		return np.argmax(self.network.forward(X, train=False), axis=1)


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

	# Create and train the MLP model
	print(f"Training MLP {dataset} model...")
	model = MLP()
	model.fit(X_train, y_train)

	# Make predictions
	print(f"Making predictions for {dataset} dataset...")
	y_train_pred = model.predict(X_train)
	y_test_pred = model.predict(X_test)

	# convert the predictions to binary
	# get the maximum value of the predictions
	mean_train = np.mean(y_train_pred)
	y_train_pred = np.where(y_train_pred > mean_train, 1, 0)
	mean_test = np.mean(y_test_pred)
	y_test_pred = np.where(y_test_pred > mean_test, 1, 0)

	# Evaluate the model
	print(f"Evaluating the model for {dataset} dataset...")
	# Training metrics
	train_accuracy = accuracy_score(y_train, y_train_pred)
	train_precision = precision_score(y_train, y_train_pred, average='weighted', zero_division=1)
	train_recall = recall_score(y_train, y_train_pred, average='weighted', zero_division=1)
	train_f1 = f1_score(y_train, y_train_pred, average='weighted', zero_division=1)

	# Test metrics
	test_accuracy = accuracy_score(y_test, y_test_pred)
	test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=1)
	test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=1)
	test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=1)

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
	if not os.path.exists(f'./tema2_{dataset}/MLP'):
		os.makedirs(f'./tema2_{dataset}/MLP')

	# Save the confusion matrix
	plt.savefig(f'./tema2_{dataset}/MLP/confusion_matrix_scikit-learn.png')

	# Save the metrics
	with open(f'./tema2_{dataset}/MLP/metrics_scikit-learn.txt', 'w') as f:
		f.write(f"Training accuracy: {train_accuracy}\n")
		f.write(f"Training precision: {train_precision}\n")
		f.write(f"Training recall: {train_recall}\n")
		f.write(f"Training f1: {train_f1}\n")
		f.write(f"Test accuracy: {test_accuracy}\n")
		f.write(f"Test precision: {test_precision}\n")
		f.write(f"Test recall: {test_recall}\n")
		f.write(f"Test f1: {test_f1}\n")



if __name__ == '__main__':
	solver('SalaryPrediction')
	solver('AVC')
