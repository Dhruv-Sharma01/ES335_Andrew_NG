"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd 
import numpy as np 

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """
    X_encoded = pd.get_dummies(X).astype(int)
    return X_encoded
    # pass

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    # Check if the series dtype is a floating point type (real/continuous data)
    if_real = pd.api.types.is_float_dtype(y) or pd.api.types.is_numeric_dtype(y)
    return if_real

    # pass


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    # Calculate the proportion of each class in the dataset
    class_probs = Y.value_counts(normalize=True)
    values =  -np.sum(class_probs * np.log2(class_probs))
    # Replace NaN values with 0
    values = np.nan_to_num(values, nan=0.0)
    return values
    # pass


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    # Calculate the proportion of each class in the dataset
    class_probs = Y.value_counts(normalize=True)
    values = 1 -np.sum(class_probs **2 )
    return values
    # pass


def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)
    """
    # Initial impurity of the dataset before splitting
    if criterion == 'gini_index':
        initial_impurity = gini_index(Y)
    elif criterion == 'entropy':
        initial_impurity = entropy(Y)
    elif criterion == 'MSE':
        initial_impurity = MSE(Y)
    # else:
    #     raise ValueError("Criterion should be one of 'gini_index', 'entropy', or 'MSE'.")
    else:
        initial_impurity = entropy(Y)

    # Calculate the weighted impurity after splitting
    unique_values = attr.unique()
    weighted_impurity = 0
    for value in unique_values:
        subset_Y = Y[attr == value]
        weight = len(subset_Y) / len(Y)
        
        if criterion == 'gini_index':
            impurity = gini_index(subset_Y)
        elif criterion == 'entropy':
            impurity = entropy(subset_Y)
        elif criterion == 'MSE':
            impurity = MSE(subset_Y)
        else:
            impurity = entropy(subset_Y)
        
        weighted_impurity += weight * impurity

    # Information Gain is the reduction in impurity
    info_gain = initial_impurity - weighted_impurity
    
    return info_gain
    # pass


def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """

    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).
    best_gain = -float("inf")
    best_attribute = None
    best_split_value = None
    
    for feature in features:
        unique_values = X[feature].unique()
        
        for value in unique_values:
            # For discrete features, split by equality to the value
            if pd.api.types.is_categorical_dtype(X[feature]) or pd.api.types.is_object_dtype(X[feature]):
                y_left = y[X[feature] == value]
                y_right = y[X[feature] != value]
            else:
                # For continuous features, split by less than or equal to the value
                y_left = y[X[feature] <= value]
                y_right = y[X[feature] > value]
            
            if len(y_left) == 0 or len(y_right) == 0:
                continue

            gain = information_gain(y, X[feature] , criterion)
            
            if gain > best_gain:
                best_gain = gain
                best_attribute = feature
                best_split_value = value
    
    return best_attribute, best_split_value
    # pass


def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """

    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.
    if pd.api.types.is_categorical_dtype(X[attribute]) or pd.api.types.is_object_dtype(X[attribute]):
        # Discrete feature split
        mask = X[attribute] == value
    else:
        # Continuous feature split
        mask = X[attribute] <= value
    
    X_left = X[mask]
    y_left = y[mask]
    X_right = X[~mask]
    y_right = y[~mask]
    
    return X_left, y_left, X_right, y_right
    # pass

def MSE(Y: pd.Series) -> float:
    mse = np.mean((Y - np.mean(Y)) ** 2)
    return mse
