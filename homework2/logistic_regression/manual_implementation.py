import numpy as np
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=10000, regularization_strength=0.1):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.regularization_strength = regularization_strength

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_cost(self, X, y, weights):
        m = len(y)
        h = self.sigmoid(np.dot(X, weights))
        regularization_term = (self.regularization_strength / (2 * m)) * np.sum(np.square(weights[1:]))
        cost = -1/m * (np.dot(y, np.log(h)) + np.dot((1 - y), np.log(1 - h))) + regularization_term
        return cost

    def gradient_descent(self, X, y, weights):
        m = len(y)
        for i in range(self.num_iterations):
            h = self.sigmoid(np.dot(X, weights))
            gradient = np.dot(X.T, (h - y)) / m
            regularization_term = (self.regularization_strength / m) * weights
            regularization_term[0] = 0
            weights -= self.learning_rate * (gradient + regularization_term)
            if i % 1000 == 0:
                cost = self.compute_cost(X, y, weights)
        return weights

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        weights = np.zeros(X.shape[1])
        self.weights = self.gradient_descent(X, y, weights)

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        probabilities = self.sigmoid(np.dot(X, self.weights))
        return probabilities >= 0.5

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

    X_train = train.drop(target, axis=1)
    y_train = train[target]

    X_test = test.drop(target, axis=1)
    y_test = test[target]

    # Convert bool columns to float64
    X_train = X_train.astype({col: 'float64' for col in X_train.select_dtypes(include='bool').columns})
    X_test = X_test.astype({col: 'float64' for col in X_test.select_dtypes(include='bool').columns})

    # Convert to numpy arrays
    X_train = X_train.values
    y_train = y_train.values
    X_test = X_test.values
    y_test = y_test.values

    # Create and train the Logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Convert the predictions to binary
    mean_train = np.mean(y_train_pred)
    y_train_pred = np.where(y_train_pred > mean_train, 1, 0)
    mean_test = np.mean(y_test_pred)
    y_test_pred = np.where(y_test_pred > mean_test, 1, 0)

    # Evaluate the model
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

    # Plot the confusion matrix
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
    if not os.path.exists(f'./tema2_{dataset}/LogisticRegression'):
        os.makedirs(f'./tema2_{dataset}/LogisticRegression')

    # Save the confusion matrix
    plt.savefig(f'./tema2_{dataset}/LogisticRegression/confusion_matrix_manual.png')

    # Save the metrics
    with open(f'./tema2_{dataset}/LogisticRegression/metrics_manual.txt', 'w') as f:
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
