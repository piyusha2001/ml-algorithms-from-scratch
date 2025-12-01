import numpy as np

# This method standardizes the features by removing the mean and scaling to unit variance.
class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        X = np.array(X, dtype=np.float64) 
        if np.any(np.isnan(X)):
            raise ValueError("Input data contains NaN values.")

        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)

        self.std_ = np.where(self.std_ == 0, 1, self.std_)

    def transform(self, X):
        X = np.array(X, dtype=np.float64)  
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_scaled):
        X_scaled = np.array(X_scaled, dtype=np.float64)  
        
        return X_scaled * self.std_ + self.mean_
