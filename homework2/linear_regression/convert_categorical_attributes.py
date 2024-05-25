import numpy as np
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report


def convert_categorical_attributes(X_train, X_test, cat_attr, header):
# Convert categorical attributes to one-hot encoded format
    cat_indices = [list(header).index(attr) for attr in cat_attr]

    categorical_data = X_train[:, cat_indices]
    categorical_df = pd.DataFrame(categorical_data, columns=cat_attr)
    XX_train = pd.get_dummies(categorical_df, prefix=cat_attr)

    categorical_data = X_test[:, cat_indices]
    categorical_df = pd.DataFrame(categorical_data, columns=cat_attr)
    XX_test = pd.get_dummies(categorical_df, prefix=cat_attr)

    XX_train = XX_train.reindex(columns=XX_test.columns, fill_value=0)
    XX_test = XX_test.reindex(columns=XX_train.columns, fill_value=0)

    # Drop original categorical columns
    X_train = np.delete(X_train, cat_indices, axis=1)
    X_test = np.delete(X_test, cat_indices, axis=1)
    header = np.delete(header, cat_indices)

    # Concatenate the original array with the new one-hot encoded columns
    X_train = np.concatenate((X_train, XX_train.values), axis=1)
    X_test = np.concatenate((X_test, XX_test.values), axis=1)
    header = np.concatenate((header, XX_train.columns))


    return X_train, X_test, header


if __name__ == "__main__":
    # get the type of dataset
    if len(sys.argv) < 2:
        print("Usage: python convert_categorical_attributes.py <dataset>")
        sys.exit(1)
    dataset = sys.argv[1]
    train_file = f'./tema2_{dataset}/preprocessed_correlated_{dataset}_train_standardized.csv'
    test_file = f'./tema2_{dataset}/preprocessed_correlated_{dataset}_test_standardized.csv'
    
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print("Please standardize the dataset first")
        sys.exit(1)

    categorical_attributes = []
    with open(f'./tema2_{dataset}/categorical_attributes.txt') as f:
        categorical_attributes = f.read().splitlines()
    
    # check if all the categorical attributes are in the dataset
    header = pd.read_csv(train_file).columns
    for cat_attr in categorical_attributes:
        if cat_attr not in header:
            categorical_attributes.remove(cat_attr)
    # remove the target attribute
    categorical_attributes.remove(header[-1])
    
    # load the datasets
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    # extract the features and the target
    X_train = train.iloc[:, :-1].values
    y_train = train.iloc[:, -1].values

    X_test = test.iloc[:, :-1].values
    y_test = test.iloc[:, -1].values

    # convert categorical attributes to one-hot
    X_train, X_test, new_header = convert_categorical_attributes(X_train, X_test, categorical_attributes, header[:-1])

    # convert the target to a binary format if needed
    if dataset == 'SalaryPrediction':
        y_train = np.where(y_train == '>50K', 1, 0)
        y_test = np.where(y_test == '>50K', 1, 0)

    # save the new datasets
    new_train_file = train_file.replace('.csv', '_one_hot.csv')
    new_test_file = test_file.replace('.csv', '_one_hot.csv')

    # add the target column
    X_train = np.column_stack((X_train, y_train))
    X_test = np.column_stack((X_test, y_test))
    new_header = np.append(new_header, header[-1])

    pd.DataFrame(X_train, columns=new_header).to_csv(new_train_file, index=False)
    pd.DataFrame(X_test, columns=new_header).to_csv(new_test_file, index=False)
    print(f"New datasets saved in {new_train_file} and {new_test_file}")

