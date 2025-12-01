import numpy as np

class SVM:
    def __init__(self, C=1.0, learning_rate=0.01, num_iterations=1000):
        self.C = C
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.w = None  
        self.b = None  

    def fit(self, X, y):
        m, n = X.shape  # m = number of samples, n = number of features

        self.w = np.zeros(n)
        self.b = 0

        # Gradient Descent Loop
        for i in range(self.num_iterations):
            decision_boundary = np.dot(X, self.w) + self.b

            hinge_loss = np.maximum(0, 1 - y * decision_boundary)

            dw = self.w - self.C * np.dot(X.T, y * (hinge_loss > 0)) / m  # Regularization term + hinge loss
            db = -self.C * np.sum(y * (hinge_loss > 0)) / m

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            if i % 100 == 0:
                loss = 0.5 * np.dot(self.w, self.w) + self.C * np.sum(np.maximum(0, 1 - y * decision_boundary))
                print(f"Iteration {i}: Loss = {loss}")

    def predict(self, X):
        decision_boundary = np.dot(X, self.w) + self.b
        return np.sign(decision_boundary)  # +1 or -1
