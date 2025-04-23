import numpy as np

#Use Z-Score Standardization
# This method standardizes the features by removing the mean and scaling to unit variance.
class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        # Ensure X is a numpy array and handle only numeric columns
        X = np.array(X, dtype=np.float64)  # Force X to be float type

        # Check if any NaN values exist in X
        if np.any(np.isnan(X)):
            raise ValueError("Input data contains NaN values.")

        # Compute mean and std for each feature (only numeric columns)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)

        # Handle zero variance by setting std to 1 if it's 0
        self.std_ = np.where(self.std_ == 0, 1, self.std_)

    def transform(self, X):
        # Ensure X is a numpy array and handle only numeric columns
        X = np.array(X, dtype=np.float64)  # Force X to be float type

        # Standardize using mean and std
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_scaled):
        # Ensure X_scaled is a numpy array
        X_scaled = np.array(X_scaled, dtype=np.float64)  # Force X_scaled to be float type
        
        # Revert the scaling
        return X_scaled * self.std_ + self.mean_
