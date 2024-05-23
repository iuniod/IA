import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import numpy as np
import sys
import os

def numerical_features(file_name):
	file = pd.read_csv(file_name)
	cols = file.select_dtypes(include=['number']).columns

	# Create a correlation matrix
	corr_matrix = file[cols].corr()

	# Generate a heatmap for the correlation matrix
	plt.figure()
	plt.matshow(corr_matrix, cmap='coolwarm')
	plt.colorbar()
	plt.xticks(np.arange(len(cols)), cols, rotation=45)
	plt.yticks(np.arange(len(cols)), cols)
	plt.title(f'Correlation Matrix for {os.path.basename(file_name).split(".")[0]} dataset')
	# make the plot bigger
	fig = plt.gcf()
	fig.set_size_inches(15, 10.5)
	plt.savefig(f'plots/correlation_matrix_{os.path.basename(file_name).split(".")[0]}.png', dpi=300)
	plt.close()

def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def cramers_v_matrix(df):
    features = df.columns
    matrix = pd.DataFrame(index=features, columns=features, dtype=float)
    for feature_1 in features:
        for feature_2 in features:
            if feature_1 == feature_2:
                matrix.loc[feature_1, feature_2] = 1.0
            else:
                confusion_matrix = pd.crosstab(df[feature_1], df[feature_2])
                matrix.loc[feature_1, feature_2] = cramers_v(confusion_matrix)
    return matrix

def cathegorical_features(file_name):
	file = pd.read_csv(file_name)
	cols = file.select_dtypes(include=['object', 'category']).columns

	corelation = cramers_v_matrix(file[cols])

	# print(f'Cramer\'s V matrix:\n{corelation}')
	# print high corelation values in a plot
	plt.figure()
	plt.matshow(corelation, cmap='coolwarm')
	plt.colorbar()
	plt.xticks(np.arange(len(cols)), cols, rotation=45)
	plt.yticks(np.arange(len(cols)), cols)
	plt.title(f'Cramer\'s V Matrix for {os.path.basename(file_name).split(".")[0]} dataset')
	# make the plot bigger
	fig = plt.gcf()
	fig.set_size_inches(15, 10.5)
	plt.savefig(f'plots/cramer_v_matrix_{os.path.basename(file_name).split(".")[0]}.png', dpi=300)
	plt.close()

	for i in range(len(cols)):
		for j in range(i+1, len(cols)):
			if corelation.iloc[i, j] > 0.7:
				print(f'High correlation between {cols[i]} and {cols[j]}')

if __name__ == "__main__":
    # Salary Prediction datasets
	print('Salary Prediction datasets')
	numerical_features('./tema2_SalaryPrediction/SalaryPrediction_full.csv')
	cathegorical_features('./tema2_SalaryPrediction/SalaryPrediction_full.csv')
	
	# Stroke Prediction datasets
	print('Stroke Prediction datasets')
	numerical_features('./tema2_AVC/AVC_full.csv')
	cathegorical_features('./tema2_AVC/AVC_full.csv')