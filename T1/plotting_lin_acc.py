import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Path to the data directories
test_path = 'Combined/Train'  # Update with the correct path
# Define activity classes and corresponding folders
activities = ['LAYING', 'SITTING', 'STANDING', 'WALKING', 'WALKING_DOWNSTAIRS', 'WALKING_UPSTAIRS']

columns = ['accx', 'accy', 'accz']

# Function to calculate linear acceleration
def calculate_linear_acceleration(df):
    return np.sqrt(df['accx']**2 + df['accy']**2 + df['accz']**2)

# Create a subplot with 6 columns (one for each activity)
fig, axes = plt.subplots(1, 6, figsize=(24, 6))

for i, activity in enumerate(activities):
    # Get one sample file from each activity class
    activity_path = os.path.join(test_path, activity)
    sample_file = os.listdir(activity_path)[1]  # Select the first file
    sample_data = pd.read_csv(os.path.join(activity_path, sample_file))

    # Calculate linear acceleration and add it as a new column
    sample_data['linear_acceleration'] = calculate_linear_acceleration(sample_data)

    # Plot the waveform for accx, accy, accz, and linear acceleration
    
    axes[i].plot(sample_data['linear_acceleration'], label='linear_acceleration', color='purple')
    axes[i].set_title(activity)
    axes[i].set_xlabel('Time')
    axes[i].set_ylabel('Acceleration')
    axes[i].legend()

plt.tight_layout()
plt.show()
