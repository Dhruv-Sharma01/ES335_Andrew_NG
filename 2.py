import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Path to the data directories
test_path = r'Combined\Train'  # Update with the correct path
activities = ['LAYING', 'SITTING', 'STANDING', 'WALKING', 'WALKING_DOWNSTAIRS', 'WALKING_UPSTAIRS']

# Initialize list to store acceleration data and activity labels
acc_data = []
activity_labels = []

# Calculate and store acceleration components for each activity
for activity in activities:
    activity_path = os.path.join(test_path, activity)
    sample_file = os.listdir(activity_path)[0]  # Select the first file
    sample_data = pd.read_csv(os.path.join(activity_path, sample_file))
    
    # Store the acceleration components (accx, accy, accz) and activity labels
    acc_data.append(sample_data[['accx', 'accy', 'accz']].values)
    activity_labels.extend([activity] * len(sample_data))

# Concatenate all acceleration data into a single array
acc_data = np.vstack(acc_data)

# Standardize the data before applying PCA
scaler = StandardScaler()
acc_data_scaled = scaler.fit_transform(acc_data)

# Apply PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(acc_data_scaled)

# Create a DataFrame for PCA results and activity labels
df_pca = pd.DataFrame({
    'PCA1': pca_result[:, 0],
    'PCA2': pca_result[:, 1],
    'activity': activity_labels
})

# Plot PCA results
plt.figure(figsize=(10, 7))
for activity in df_pca['activity'].unique():
    subset = df_pca[df_pca['activity'] == activity]
    plt.scatter(subset['PCA1'], subset['PCA2'], label=activity, alpha=0.6)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Accelerometer Data')
plt.legend()
plt.show()
