import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, learning_rate=0.001, epochs=2000, lambda_=0.01):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lambda_ = lambda_  # Regularization strength
        self.weights = None
        self.bias = None
        self.losses = []
    
    def sigmoid(self, z):
        """Sigmoid function with overflow protection."""
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def cost(self, X, y):
        """Compute the cost function with L2 regularization."""
        m = X.shape[0]
        y_pred = self.sigmoid(np.dot(X, self.weights) + self.bias)
        cost = -(1/m) * np.sum(
            y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15)
        )
        reg_cost = (self.lambda_ / (2 * m)) * np.sum(self.weights ** 2)
        return cost + reg_cost
    
    def gradient_descent(self, X, y):
        """Perform one step of gradient descent with L2 regularization."""
        m = X.shape[0]
        y_pred = self.sigmoid(np.dot(X, self.weights) + self.bias)
        
        dw = (1/m) * np.dot(X.T, (y_pred - y)) + (self.lambda_ / m) * self.weights
        db = (1/m) * np.sum(y_pred - y)
        
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db
        
    def fit(self, X, y):
        """Train the model."""
        X = np.array(X)
        y = np.array(y).flatten()
        
        # Feature Standardization
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-8  # avoid division by zero
        X = (X - self.mean) / self.std

        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0
        
        print(f"Initial weights: {self.weights}, Initial bias: {self.bias}")
        
        for i in range(self.epochs):
            self.gradient_descent(X, y)
            cost = self.cost(X, y)
            self.losses.append(cost)
            
            if i % 100 == 0:
                print(f"Epoch {i} | Cost: {cost:.4f}")
        
        self.plot_loss()
    
    def predict(self, X):
        """Predict class labels for given data."""
        X = (X - self.mean) / self.std  # Standardize using training mean/std
        y_pred = self.sigmoid(np.dot(X, self.weights) + self.bias)
        return (y_pred >= 0.5).astype(int)
    
    def accuracy(self, X, y):
        """Calculate model accuracy."""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
    def plot_loss(self):
        """Plot loss curve over epochs."""
        plt.plot(range(len(self.losses)), self.losses)
        plt.title('Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Binary Cross-Entropy Loss')
        plt.grid(True)
        plt.show()
