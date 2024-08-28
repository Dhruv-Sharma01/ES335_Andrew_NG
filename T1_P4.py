import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load X_train using np.loadtxt
X_train = np.loadtxt('HAR/UCI HAR Dataset/train/X_train.txt')

# Load X_train_tsfel using np.loadtxt
X_train_tsfel = np.loadtxt('X_train_tsfel.csv', delimiter=',')

# Convert numpy arrays to DataFrames to calculate correlations
X_train_df = pd.DataFrame(X_train)
X_train_tsfel_df = pd.DataFrame(X_train_tsfel)

# Calculate the correlation matrix between the two datasets column-wise
correlation_matrix = X_train_df.corrwith(X_train_tsfel_df, axis=0)

# Since X_train and X_train_tsfel have different columns, we calculate the correlation between each pair of columns
correlation_matrix = pd.DataFrame(np.corrcoef(X_train_df.T, X_train_tsfel_df.T)[:X_train_df.shape[1], X_train_df.shape[1]:])

# Plot the heatmap of the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
plt.xlabel('TSFEL Features')
plt.ylabel('Original Features')
plt.title('Correlation between Original Features and TSFEL Features')
plt.show()
