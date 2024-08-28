import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from base import DecisionTree
from metrics import accuracy, rmse, mae, precision, recall

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

np.random.seed(42)

# Reading the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None,
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin", "car name"])

# Clean the above data by removing redundant columns and rows with junk values
# Compare the performance of your model with the decision tree module from scikit learn

# Display the first few rows
# print(data.head())
# # Check for missing values
# print(data.isnull().sum())
# Summary statistics
# print(data.describe())
# # Data types
# print(data.dtypes)


data.dropna(inplace=True)
# print(data)
y = data['mpg']
# print(y)
X = data.drop('mpg', axis=1)
# print(X)
# one hot encoding
X = pd.get_dummies(X, columns=['car name'], drop_first=True)
X.replace('?', 0, inplace=True)

#get_data
y_dataset = y.to_list()
X_dataset = X.values.tolist()
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, test_size=0.3, random_state=42)


################################################################################################################33

# Initialize and train the model
dt_regressor = DecisionTreeRegressor(random_state=42)
dt_regressor.fit(X_train, y_train)

# Predict on the test set
y_pred = dt_regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse1 = mse ** 0.5

print(f"Root Mean Squared Error using inbuilt Decision Tree Regressor: {rmse1}")


################################################################################################################ 
# Convert X to a DataFrame
X_train_df = pd.DataFrame(X_train)
X_test_df = pd.DataFrame(X_test)
# Convert y to a Series
y_train_series = pd.Series(y_train)
y_test_series = pd.Series(y_test)

for criteria in ["MSE"]:
    tree = DecisionTree(criterion=criteria)  # Split based on Inf. Gain
    tree.fit(X_train_df, y_train_series)
    y_pred = tree.predict(X_test_df)
    # tree.plot()
    # print("plot hogya")
#     print("Criteria :", criteria)
    print("Root Mean Squared Error using Decision Tree I made: ", rmse(y_pred, y_test_series))
#     mse = mean_squared_error(y_test, y_pred)
#     rmse = mse ** 0.5
       
#     print("MAE: ", mae(y_pred, y_test_series))
