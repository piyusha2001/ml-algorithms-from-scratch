import numpy as np


def train_test_split_custom(X, y, test_size=0.2, random_seed=42, shuffle=True):
    """
    Splits the data into training and testing sets.

    Parameters:
    - X: Features data (numpy array or pandas DataFrame)
    - y: Target data (numpy array or pandas DataFrame)
    - test_size: Proportion of the data to be used for testing (default 0.2)
    - random_seed: Seed for random number generator (default 42)
    - shuffle: Whether to shuffle the data before splitting (default True)

    Returns:
    - X_train, X_test, y_train, y_test: Split datasets
    """
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # Shuffle the data if required
    if shuffle:
        indices = np.random.permutation(len(X))
    else:
        indices = np.arange(len(X))
    
    # Split indices based on test size
    test_size = int(len(X) * test_size)
    train_size = len(X) - test_size

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # Return the split data
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]