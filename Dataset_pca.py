# import numpy as np
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt

# # Paths to the files
# X_train_path = 'HAR/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset/train\X_train.txt'
# y_train_path = 'HAR/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset/train\y_train.txt'

# # Load the data
# X_train = np.loadtxt(X_train_path).astype(float)
# y_train = np.loadtxt(y_train_path).astype(int)

# # Standardize the data
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)

# # Apply PCA
# pca = PCA(n_components=2)  # Reduce to 2 components for visualization
# X_pca = pca.fit_transform(X_train_scaled)

# # Visualize the first two principal components
# plt.figure(figsize=(10, 7))
# for i, label in enumerate(np.unique(y_train)):
#     plt.scatter(X_pca[y_train == label, 0], X_pca[y_train == label, 1], label=f'Activity {label}')

# plt.title('PCA of X_train')
# plt.xlabel('First Principal Component')
# plt.ylabel('Second Principal Component')
# plt.legend()
# plt.show()

# # Optionally, you can print the explained variance ratio to see how much variance each principal component explains
# explained_variance = pca.explained_variance_ratio_
# print(f'Explained variance by the first two components: {explained_variance[:2]}')

#######################################################################################################################################################
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Paths to the files
X_train_path = 'HAR/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset/train/X_train.txt'
y_train_path = 'HAR/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset/train/y_train.txt'

# Load the data
X_train = np.loadtxt(X_train_path).astype(float)
y_train = np.loadtxt(y_train_path).astype(int)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Apply PCA to reduce to the first principal component
pca = PCA(n_components=1)  # Keep only the first 3 components
X_train_pca = pca.fit_transform(X_train_scaled)

# Save the PCA-transformed data (only numbers, no column names or labels)
output_path = 'X_train_pca.txt'
np.savetxt(output_path, X_train_pca, fmt='%f')  # fmt='%f' ensures floating-point numbers are saved

print(f"PCA-transformed data saved to {output_path}")
