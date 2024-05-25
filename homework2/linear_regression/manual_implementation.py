import numpy as np
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class LinearRegression(object):
    def fit(self, X, t):
        N, D = X.shape
        X_transpose = np.transpose(X)
        self.w = np.linalg.pinv(X_transpose @ X) @ X_transpose @ t
        self.var = np.mean(np.square(X @ self.w - t))

    def predict(self, X, return_std=False):
        N, D = X.shape
        y = X @ self.w
        
        if return_std:
            y_std = np.sqrt(self.var)
            return y, y_std

        return y


class RidgeRegression(LinearRegression):
    def __init__(self, alpha:float=1.):
        super(RidgeRegression, self).__init__()
        self.alpha = alpha
        

    def fit(self, X:np.ndarray, t:np.ndarray):
        N, D = X.shape
        X_transpose = np.transpose(X)
        self.w = np.linalg.inv(X_transpose @ X + self.alpha * np.eye(D)) @ X_transpose @ t

        self.var = np.mean(np.square(X @ self.w - t))


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

    # Create and train the linear regression model
    model = RidgeRegression(alpha=1.75)
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
    if not os.path.exists(f'./tema2_{dataset}/LinearRegression'):
        os.makedirs(f'./tema2_{dataset}/LinearRegression')

    # Save the confusion matrix
    plt.savefig(f'./tema2_{dataset}/LinearRegression/confusion_matrix_manual.png')

    # Save the metrics
    with open(f'./tema2_{dataset}/LinearRegression/metrics_manual.txt', 'w') as f:
        f.write(f"Training accuracy: {train_accuracy}\n")
        f.write(f"Training precision: {train_precision}\n")
        f.write(f"Training recall: {train_recall}\n")
        f.write(f"Training f1: {train_f1}\n")
        f.write(f"Test accuracy: {test_accuracy}\n")
        f.write(f"Test precision: {test_precision}\n")
        f.write(f"Test recall: {test_recall}\n")
        f.write(f"Test f1: {test_f1}\n")
