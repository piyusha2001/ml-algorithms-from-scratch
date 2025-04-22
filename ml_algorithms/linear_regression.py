# Linear Regression Implementation in Python
# This code implements a simple linear regression model from scratch.
import numpy as np
import matplotlib.pyplot as plt

class UnivariateLinearRegression:
    def __init__(self):
        # Initialize the model parameters (weight and bias)
        self.weight = 0.0  # slope (w)
        self.bias = 0.0    # intercept (b)

    def predict(self, X):
        # Make predictions using the linear regression equation: y = wx + b
        X = np.array(X)
        return self.weight * X + self.bias
    
    def cost(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        predictions = self.predict(X)
        errors = y - predictions
        mse = np.mean(errors ** 2)
        
        return mse
   

    def fit(self, X, y, learning_rate=0.01, epochs=1000, tolerance=1e-6):
        X = np.array(X)
        y = np.array(y)
        n = len(X)

        prev_loss = float('inf')  # Initialize loss as infinity
        loss_values = []  # List to track loss over epochs

        for epoch in range(epochs):
            predictions = self.predict(X)  # Make predictions with current weight and bias
            errors = y - predictions  # Calculate errors (difference between predicted and actual)

            # Compute gradients (dw and db)
            dw = -2 * np.dot(errors, X) / n  # Gradient for weight
            db = -2 * np.mean(errors)  # Gradient for bias

            # Update parameters
            self.weight -= learning_rate * dw  # Update weight
            self.bias -= learning_rate * db  # Update bias

            # Calculate and store loss (cost)
            loss = self.cost(X, y)  # Calculate the cost using the updated parameters
            loss_values.append(loss)

            # Print loss every 100 epochs for tracking
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.2f}")

            # Early stopping: if the loss improvement is smaller than the tolerance, stop
            if abs(prev_loss - loss) < tolerance:
                print(f"Early stopping at epoch {epoch}, Loss = {loss:.2f}")
                break

            prev_loss = loss  # Store the loss for the next iteration

        # Plot the loss over epochs to visualize convergence
        plt.plot(loss_values)
        plt.title("Loss over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show()

    def plot_fit(self, X, y):
        # Visualize the regression line and data points
        X = np.array(X)
        y = np.array(y)
        plt.scatter(X, y, color='blue', label='Data points')
        y_pred = self.predict(X)
        plt.plot(X, y_pred, color='red', label='Regression line')
        plt.title("Linear Regression Fit")
        plt.xlabel("Years of Experience")
        plt.ylabel("Salary")
        plt.legend()
        plt.grid(True)
        plt.show()

    def print_parameters(self):
        # Print learned parameters
        print(f"Learned weight (slope): {self.weight:.4f}")
        print(f"Learned bias (intercept): {self.bias:.4f}")

class MultipleLinearRegression:
    def __init__(self):
        self.weights = None  

    def predict(self, X):
        X = np.array(X)
        return np.dot(X, self.weights)
    
    def cost(self, X, y):
        predictions = self.predict(X)
        errors = predictions - y
        return np.mean(errors ** 2)  # MSE

    def fit(self, X, y, learning_rate=0.01, epochs=1000, tolerance=1e-6):
        X = np.array(X)
        y = np.array(y).flatten()  # ✅ Ensure y is 1D for correct dot product
        n_samples, n_features = X.shape

        # We've already added a column of ones to X before calling fit()
        # e.g., X = np.c_[np.ones((X.shape[0], 1)), X]
        # So, the first column of X corresponds to the bias term.
        # This means weights[0] acts as the bias, and it gets updated just like any other weight.

        # Initialize weights (including bias)
        self.weights = np.zeros(n_features)  # Shape: (n_features,)

        prev_loss = float('inf')
        loss_values = []

        for epoch in range(epochs):
            predictions = self.predict(X)  # Shape: (n_samples,)
            errors = predictions - y       # Shape: (n_samples,)

            # Gradient descent update rule
            gradients = (2 / n_samples) * np.dot(X.T, errors)  # Shape: (n_features,)
            self.weights -= learning_rate * gradients  # Update weights

            # Compute mean squared error
            loss = self.cost(X, y)
            loss_values.append(loss)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}")

            # Early stopping
            if abs(prev_loss - loss) < tolerance:
                print(f"Early stopping at epoch {epoch}, Loss = {loss:.4f}")
                break

            prev_loss = loss

        # Plot loss over epochs
        plt.plot(loss_values)
        plt.title("Loss over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.grid(True)
        plt.show()

    def print_parameters(self):
        print("Learned weights:", self.weights)

    def r2_score(self, X, y):
        y = y.flatten()
        predictions = self.predict(X)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - predictions) ** 2)
        return 1 - (ss_residual / ss_total)
    
    def plot_predictions(self, X, y):
        y = y.flatten()
        y_pred = self.predict(X)
        plt.scatter(y, y_pred, color='blue')
        plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--')  # Add a line for perfect predictions
        plt.xlabel('Actual values')
        plt.ylabel('Predicted values')
        plt.title('Actual vs Predicted')
        plt.grid(True)
        plt.show()
    
    def predict_house_price(model, scaler, features: list):
        input_data = np.array([features])
        input_scaled = scaler.transform(input_data)
        input_scaled = np.c_[np.ones(input_scaled.shape[0]), input_scaled]
        prediction = model.predict(input_scaled)
        return prediction[0]
