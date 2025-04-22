import numpy as np

#Use Z-Score Standardization
# This method standardizes the features by removing the mean and scaling to unit variance.
class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        # Compute mean and std for each feature
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)

    def transform(self, X):
        # Standardize using mean and std
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_scaled):
        # Revert the scaling
        return X_scaled * self.std_ + self.mean_
