import numpy as np


def train_test_split_custom(X, y, test_size=0.2, random_seed=42, shuffle=True):

    np.random.seed(random_seed)

    if shuffle:
        indices = np.random.permutation(len(X))
    else:
        indices = np.arange(len(X))
    
    test_size = int(len(X) * test_size)
    train_size = len(X) - test_size

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]