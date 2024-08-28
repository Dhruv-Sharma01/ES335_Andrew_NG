from typing import Union
import pandas as pd
import numpy as np


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size
    # TODO: Write here
    assert y_hat.size == y.size, "Predicted and true labels must have the same size"
    
    correct_predictions = (y_hat == y).sum()
    total_predictions = y.size
    accuracy = correct_predictions / total_predictions
    
    return accuracy
    # pass


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    assert y_hat.size == y.size, "Predicted and true labels must have the same size"
    
    true_positives = ((y_hat == cls) & (y == cls)).sum()
    predicted_positives = (y_hat == cls).sum()
    
    if predicted_positives == 0:
        return 0.0  # Avoid division by zero
    
    precision = true_positives / predicted_positives
    
    return precision
    # pass


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    assert y_hat.size == y.size, "Predicted and true labels must have the same size"
    
    true_positives = ((y_hat == cls) & (y == cls)).sum()
    actual_positives = (y == cls).sum()
    
    if actual_positives == 0:
        return 0.0  # Avoid division by zero
    
    recall = true_positives / actual_positives
    
    return recall
    # pass


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    assert y_hat.size == y.size, "Predicted and true labels must have the same size"
    
    mse = ((y_hat - y) ** 2).mean()
    rmse = np.sqrt(mse)
    
    return rmse
    # pass


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    assert y_hat.size == y.size, "Predicted and true labels must have the same size"
    
    mae = (y_hat - y).abs().mean()
    
    return mae
    # pass
