import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k  # Number of neighbors to consider
    
    def euclidean_distance(self, point1, point2):
        # Calculate the Euclidean distance between two points
        return np.sqrt(np.sum((point1 - point2) ** 2))
    
    def fit(self, X_train, y_train):
        # Store the training data
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):
        # Predict the class for each test point
        y_pred = [self._predict(x) for x in X_test]
        return np.array(y_pred)
    
    def _predict(self, x):
        # Step 1: Compute distances between x and all training points
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # Step 2: Sort by distance and get the indices of the K nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Step 3: Get the labels of the K nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Step 4: Majority voting — return the most common label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]  # Return the label of the most common neighbor

    def accuracy(self, X_test, y_test):
        # Predict the labels for the test set
        y_pred = self.predict(X_test)
        
        # Calculate accuracy as the percentage of correct predictions
        accuracy = np.mean(y_pred == y_test) * 100
        return accuracy
