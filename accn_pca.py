import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Paths to the files
acc_x_path = 'HAR/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset/train/Inertial Signals/total_acc_x_train.txt'
acc_y_path = 'HAR/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset/train/Inertial Signals/total_acc_y_train.txt'
acc_z_path = 'HAR/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset/train/Inertial Signals/total_acc_z_train.txt'
y_train_path = 'HAR/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset/train/y_train.txt'

# Load the data
acc_x = np.loadtxt(acc_x_path).astype(float)
acc_y = np.loadtxt(acc_y_path).astype(float)
acc_z = np.loadtxt(acc_z_path).astype(float)
y_train = np.loadtxt(y_train_path).astype(int)

# Calculate the total acceleration
total_acc = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

# Standardize the data
scaler = StandardScaler()
total_acc_scaled = scaler.fit_transform(total_acc)

# Apply PCA to reduce to 2 principal components
pca = PCA(n_components=2)
total_acc_pca = pca.fit_transform(total_acc_scaled)

# Create a DataFrame for plotting
df_pca = pd.DataFrame(data=total_acc_pca, columns=['PC1', 'PC2'])
df_pca['Activity'] = y_train

# Define activity labels
activity_labels = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING"
}

# Map the activity numbers to labels
df_pca['Activity_Label'] = df_pca['Activity'].map(activity_labels)

# Plotting the scatter plot
plt.figure(figsize=(10, 7))
for label, activity in activity_labels.items():
    indices = df_pca['Activity'] == label
    plt.scatter(df_pca.loc[indices, 'PC1'], 
                df_pca.loc[indices, 'PC2'], 
                label=activity, s=20)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Total Acceleration Data')
plt.legend()
plt.show()
output_path = 'total_acc_pca.txt'
np.savetxt(output_path, total_acc_pca, delimiter=',')
print(f"PCA-transformed data saved to {output_path}")