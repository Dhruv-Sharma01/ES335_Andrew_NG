import numpy as np

# Path to the X_train.txt file
file_path = 'HAR/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset/train/X_train.txt'

# Read the file
data = np.loadtxt(file_path)
print(data.shape)
# Print the shape of the data
print("Shape of the data:", data.shape)

# Extract the first time series vector
first_time_series_vector = data[0, :]
print("First time series vector:", first_time_series_vector)
print(len(first_time_series_vector))
