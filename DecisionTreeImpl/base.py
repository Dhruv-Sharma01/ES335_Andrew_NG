"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal,Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import one_hot_encoding, check_ifreal, entropy, gini_index, information_gain, opt_split_attribute, split_data

np.random.seed(42)


@dataclass
class DecisionTreeNode:
    attribute: Optional[str] = None  # The feature used for splitting
    split_value: Any = None          # The value of the feature used for splitting
    left: Optional['DecisionTreeNode'] = None  # Left subtree
    right: Optional['DecisionTreeNode'] = None  # Right subtree
    prediction: Any = None  # The prediction if this is a leaf node

class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None  # Root of the tree
        # print(criterion)
        # print(type(criterion))
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """

        # If you wish your code can have cases for different types of input and output data (discrete, real)
        # Use the functions from utils.py to find the optimal attribute to split upon and then construct the tree accordingly.
        # You may(according to your implemetation) need to call functions recursively to construct the tree. 

        # pass
        def _fit(X, y, depth):
            if depth == self.max_depth or y.nunique() == 1:
                # Create a leaf node
                if check_ifreal(y):
                    prediction = y.mean()  # For regression
                else:
                    prediction = y.mode()[0]  # For classification
                return DecisionTreeNode(prediction=prediction)

            # Find the best attribute to split upon
            best_attr, best_val = opt_split_attribute(X, y, self.criterion, features=X.columns)

            if best_attr is None:
                # If no valid split is found, create a leaf node
                if check_ifreal(y):
                    prediction = y.mean()  # For regression
                else:
                    prediction = y.mode()[0]  # For classification
                return DecisionTreeNode(prediction=prediction)

            # Split the data
            X_left, y_left, X_right, y_right = split_data(X, y, best_attr, best_val)

            # Create a decision node
            left_subtree = _fit(X_left, y_left, depth + 1)
            right_subtree = _fit(X_right, y_right, depth + 1)

            return DecisionTreeNode(attribute=best_attr, split_value=best_val, left=left_subtree, right=right_subtree)

        # Start the recursive fitting process
        self.root = _fit(X, y, 0)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """

        # Traverse the tree you constructed to return the predicted values for the given test inputs.
        def _predict(sample, node):
            if node.prediction is not None:
                return node.prediction
            if pd.api.types.is_categorical_dtype(X[node.attribute]) or pd.api.types.is_object_dtype(X[node.attribute]):
                if sample[node.attribute] == node.split_value:
                    return _predict(sample, node.left)
                else:
                    return _predict(sample, node.right)
            else:
                if sample[node.attribute] <= node.split_value:
                    return _predict(sample, node.left)
                else:
                    return _predict(sample, node.right)

        return X.apply(lambda row: _predict(row, self.root), axis=1)

        # pass

    def plot(self, depth= 0, node=None) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        if node is None:
            node = self.root
        # print(f"yoooooooooo_depth")
        # print(depth)
        # print(f"yessssssss")
        if node.prediction is not None:
            print(f"{' ' * depth * 4}Predict: {node.prediction}")
        else:
            print(f"{' ' * depth * 4}?({node.attribute} <= {node.split_value})")
            print(f"{' ' * depth * 4}Y:")
            self.plot(depth + 1, node.left)
            print(f"{' ' * depth * 4}N:")
            self.plot(depth + 1, node.right)
        
        # pass
