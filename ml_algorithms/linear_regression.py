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

