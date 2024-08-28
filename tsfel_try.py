import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tsfel

# Paths to the files
acc_x_path_train = 'HAR/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset/train/Inertial Signals/total_acc_x_train.txt'
acc_y_path_train = 'HAR/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset/train/Inertial Signals/total_acc_y_train.txt'
acc_z_path_train = 'HAR/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset/train/Inertial Signals/total_acc_z_train.txt'
y_train_path = 'HAR/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset/train/y_train.txt'
# subject_train_path = 'HAR/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset/train/subject_train.txt'
acc_x_path_test = 'HAR/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset/test/Inertial Signals/total_acc_x_test.txt'
acc_y_path_test = 'HAR/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset/test/Inertial Signals/total_acc_y_test.txt'
acc_z_path_test = 'HAR/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset/test/Inertial Signals/total_acc_z_test.txt'
y_test_path = 'HAR/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset/test/y_test.txt'
# subject_test_path = 'HAR/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset/train/subject_test.txt'
acc_x_train = np.loadtxt(acc_x_path_train).astype(float)
acc_y_train = np.loadtxt(acc_y_path_train).astype(float)
acc_z_train = np.loadtxt(acc_z_path_train).astype(float)
acc_x_test = np.loadtxt(acc_x_path_test).astype(float)
acc_y_test = np.loadtxt(acc_y_path_test).astype(float)
acc_z_test = np.loadtxt(acc_z_path_test).astype(float)
y_train = np.loadtxt(y_train_path).astype(int)
y_test = np.loadtxt(y_test_path).astype(int)
linear_acceleration_train=np.sqrt(acc_x_train**2 + acc_y_train**2 + acc_z_train**2)
linear_acceleration_test=np.sqrt(acc_x_test**2 + acc_y_test**2 + acc_z_test**2)
X_train_tsfel = np.zeros((len(linear_acceleration_train),4))
X_test_tsfel = np.zeros((len(linear_acceleration_test),4))
features=["abs_energy","slope","negative_turning","human_range_energy"]

for i in range(len(linear_acceleration_train)):
           
       X_train_tsfel[i][0]= np.array(tsfel.feature_extraction.features.abs_energy(linear_acceleration_train[i]))
       X_train_tsfel[i][1]= np.array(tsfel.feature_extraction.features.slope(linear_acceleration_train[i]))
       X_train_tsfel[i][2]= np.array(tsfel.feature_extraction.features.negative_turning(linear_acceleration_train[i]))
       X_train_tsfel[i][3]= np.array(tsfel.feature_extraction.features.human_range_energy(linear_acceleration_train[i],fs=50))
    
for i in range(len(linear_acceleration_test)):
    
       X_test_tsfel[i]=np.array(tsfel.feature_extraction.features.abs_energy(linear_acceleration_test[i]))
print(X_train_tsfel[0])
print("X_train_features shape:", X_train_tsfel.shape)
print("X_test_features shape:", X_test_tsfel.shape)

df_train = pd.DataFrame(X_train_tsfel)
df_test = pd.DataFrame(X_test_tsfel)

# Define paths for saving the CSV files
train_csv_path = 'X_train_tsfel.csv'
test_csv_path = 'X_test_tsfel.csv'

# Save to CSV
df_train.to_csv(train_csv_path, index=False)
df_test.to_csv(test_csv_path, index=False)

