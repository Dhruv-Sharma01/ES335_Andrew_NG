import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.tree import DecisionTreeClassifier


np.random.seed(42)

# Code given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
plt.scatter(X[:, 0], X[:, 1], c=y)

# Write the code for Q2 a) and b) below. Show your results.

# q2(a)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert X to a DataFrame
X_train_df = pd.DataFrame(X_train, columns=['Feature1', 'Feature2'])
X_test_df = pd.DataFrame(X_test, columns=['Feature1', 'Feature2'])
# Convert y to a Series
y_train_series = pd.Series(y_train, name='y_label')
y_test_series = pd.Series(y_test, name='y_label')

# print(X)

# print(X_train_df, X_test_df)

# test on train data

# for criteria in ["gini_index"]:
#     tree = DecisionTree(criterion=criteria)  # Split based on Inf. Gain
#     tree.fit(X_train_df, y_train_series)
#     y_pred = tree.predict(X_train_df)
#     tree.plot()
#     # print("plot hogya")
#     print("Criteria :", criteria)
#     print("RMSE: ", rmse(y_pred, y_train_series))
#     print("MAE: ", mae(y_pred, y_train_series))

# print("Separate analysis yo yo yo")

# test on test data

for criteria in ["entropy"]:
    tree = DecisionTree(criterion=criteria)  # Split based on Inf. Gain
    tree.fit(X_train_df, y_train_series)
    y_pred = tree.predict(X_test_df)
    tree.plot()
    # print("plot hogya")
    print("Criteria :", criteria)
    print("RMSE: ", rmse(y_pred, y_test_series))
    print("MAE: ", mae(y_pred, y_test_series))
    for cls in y_train_series.unique():
        print("Precision: ", precision(y_pred, y_test_series, cls))
        print("Recall: ", recall(y_pred, y_test_series, cls))
    

#######################################################3

# q2(b)

# Initialize the decision tree classifier without setting a max_depth
dt = DecisionTreeClassifier(random_state=1)

# Define 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=1)

# Define a parameter grid to search for the optimal depth
param_grid = {'max_depth': np.arange(1, 21)}

# Set up the grid search with cross-validation
grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=kf, scoring='accuracy')

# Perform nested cross-validation
nested_scores = cross_val_score(grid_search, X, y, cv=kf)

# Fit the model using the entire dataset to find the best depth
grid_search.fit(X, y)
best_depth = grid_search.best_params_['max_depth']

print(f"Nested CV accuracy: {nested_scores.mean():.4f} ± {nested_scores.std():.4f}")
print(f"Optimal depth found: {best_depth}")
