import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the feature data
X_train_tsfel = pd.read_csv('X_train_tsfel.csv')

# Load the activity labels (assuming the labels are stored in a CSV file too)
y_train = pd.read_csv('HAR/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset/train/y_train.txt').values.flatten()

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train_tsfel)
# Apply PCA to reduce dimensions to 2
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
# Plot the two principal components with activity labels
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap='viridis', edgecolor='k', alpha=0.7, s=50)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of HAR Dataset')
plt.colorbar(scatter, label='Activity Label')
plt.show()
