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

# Create a subplot with 2 rows and 3 columns (one for each activity)
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)

# Flatten the axes array for easier iteration
axes = axes.flatten()

for i, activity in enumerate(activities):
    # Get one sample file from each activity class
    activity_path = os.path.join(test_path, activity)
    sample_file = os.listdir(activity_path)[1]  # Select the first file
    sample_data = pd.read_csv(os.path.join(activity_path, sample_file))

    # Calculate linear acceleration and add it as a new column
    sample_data['linear_acceleration'] = calculate_linear_acceleration(sample_data)

    # Plot the waveform for linear acceleration
    axes[i].plot(sample_data['linear_acceleration'], label='Linear Acceleration', color='purple', linewidth=1.5)
    
    # Set titles and labels
    axes[i].set_title(activity, fontsize=14, weight='bold')
    axes[i].set_xlabel('Time', fontsize=12)
    if i % 3 == 0:  # Only label y-axis on the leftmost plots
        axes[i].set_ylabel('Acceleration', fontsize=12)
    
    # Add grid lines
    axes[i].grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Add legend
    axes[i].legend(fontsize=10)
    
    # Remove spines for a cleaner look
    axes[i].spines['top'].set_visible(False)
    axes[i].spines['right'].set_visible(False)

# Improve the layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.suptitle('Linear Acceleration Across Different Activities', fontsize=16, weight='bold')

plt.show()
