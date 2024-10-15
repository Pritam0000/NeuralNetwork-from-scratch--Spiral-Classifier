import pandas as pd
import numpy as np

def load_spiral_data(filepath):
    """
    Load spiral data from CSV file.
    
    Args:
    filepath (str): Path to the CSV file.
    
    Returns:
    tuple: (X, y) where X is the feature matrix and y is the target vector.
    """
    df = pd.read_csv(filepath)
    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].to_numpy()
    return X, y