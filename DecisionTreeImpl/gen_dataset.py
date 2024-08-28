from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c=y)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Scatter plot of the generated data")
plt.show() 


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from base import DecisionTree
from metrics import accuracy, rmse, mae, precision, recall

np.random.seed(42)
# Test case 1
# Real Input and Real Output

# N = 30
# P = 5
# X = pd.DataFrame(np.random.randn(N, P))
# y = pd.Series(np.random.randn(N))

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

 
X1_new =  X_test_df['Feature1'].tolist()
X2_new =  X_test_df['Feature2'].tolist()
# print(X_new)
plt.scatter(X1_new, X2_new, c=y_pred.to_list())
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Scatter plot of the generated data")
plt.show() 

plt.scatter(X1_new, X2_new, c=y_test_series.to_list())
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Scatter plot of the generated data")
plt.show() 