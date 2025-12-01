import numpy as np
import matplotlib.pyplot as plt

class UnivariateLinearRegression:
    def __init__(self):
        self.weight = 0.0 
        self.bias = 0.0    

    def predict(self, X):
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

        prev_loss = float('inf')  
        loss_values = []  

        for epoch in range(epochs):
            predictions = self.predict(X)  
            errors = y - predictions  

            # Compute gradients (dw and db)
            dw = -2 * np.dot(errors, X) / n 
            db = -2 * np.mean(errors)  

            self.weight -= learning_rate * dw  
            self.bias -= learning_rate * db  

            # Calculate and store loss (cost)
            loss = self.cost(X, y)  
            loss_values.append(loss)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.2f}")

            # Early stopping: if the loss improvement is smaller than the tolerance, stop
            if abs(prev_loss - loss) < tolerance:
                print(f"Early stopping at epoch {epoch}, Loss = {loss:.2f}")
                break

            prev_loss = loss  

        plt.plot(loss_values)
        plt.title("Loss over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show()

    def plot_fit(self, X, y):
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
        print(f"Learned weight (slope): {self.weight:.4f}")
        print(f"Learned bias (intercept): {self.bias:.4f}")

class MultipleLinearRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.weights = None
        self.lr = lr
        self.n_iters = n_iters 

    def predict(self, X):
        X = np.array(X)
        return np.dot(X, self.weights)
    
    def cost(self, X, y):
        predictions = self.predict(X)
        errors = predictions - y
        return np.mean(errors ** 2)  # MSE

    def fit(self, X, y, learning_rate=0.01, epochs=1000, tolerance=1e-6):
        X = np.array(X)
        y = np.array(y).flatten() 
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)  
        prev_loss = float('inf')
        loss_values = []

        for epoch in range(epochs):
            predictions = self.predict(X)  
            errors = predictions - y       

            gradients = (2 / n_samples) * np.dot(X.T, errors)  
            self.weights -= learning_rate * gradients  

            loss = self.cost(X, y)
            loss_values.append(loss)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}")

            if abs(prev_loss - loss) < tolerance:
                print(f"Early stopping at epoch {epoch}, Loss = {loss:.4f}")
                break

            prev_loss = loss

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
        plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--')  
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

class RidgeRegression(MultipleLinearRegression):
    def __init__(self, lr=0.001, n_iters=2000, lambda_=0.1):
        super().__init__(lr=lr, n_iters=n_iters)
        self.lambda_ = lambda_

    def cost(self, X, y):
        y = y.flatten()
        y_pred = self.predict(X)
        error = y_pred - y
        n_samples = X.shape[0]

        regularization = self.lambda_ * np.sum(self.weights[1:] ** 2) / n_samples

        return np.mean(error ** 2) + regularization

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).flatten()
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        loss_values = []
        prev_loss = float('inf')

        for epoch in range(self.n_iters):
            y_pred = self.predict(X)
            error = y_pred - y

            gradients = (2 / n_samples) * np.dot(X.T, error)
            gradients[1:] += (2 * self.lambda_ * self.weights[1:] / n_samples)

            self.weights -= self.lr * gradients

            loss = self.cost(X, y)
            loss_values.append(loss)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Ridge Loss = {loss:.4f}")

            if abs(prev_loss - loss) < 1e-6:
                print(f"Early stopping at epoch {epoch}, Loss = {loss:.4f}")
                break

            prev_loss = loss

        # Loss curve
        plt.plot(loss_values)
        plt.title("Ridge Loss over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.show()

class LassoRegression(MultipleLinearRegression):
    def __init__(self, lr=0.001, n_iters=1000, lambda_=1):
        super().__init__(lr=lr, n_iters=n_iters)
        self.lambda_ = lambda_

    def cost(self, X, y):
        y_pred = self.predict(X)
        error = y_pred - y
        n_samples = X.shape[0]

        # L1 regularization term (exclude bias)
        regularization = self.lambda_ * np.sum(np.abs(self.weights[1:])) / n_samples
        
        return np.mean(error ** 2) + regularization

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).flatten()
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        loss_values = []
        prev_loss = float('inf')

        for epoch in range(self.n_iters):
            y_pred = self.predict(X)
            error = y_pred - y

            gradients = (2 / n_samples) * np.dot(X.T, error)

            # Apply L1 regularization (subgradient method)
            gradients[1:] += (self.lambda_ * np.sign(self.weights[1:])) / n_samples

            self.weights -= self.lr * gradients

            loss = self.cost(X, y)
            loss_values.append(loss)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Lasso Loss = {loss:.4f}")

            if abs(prev_loss - loss) < 1e-6:
                print(f"Early stopping at epoch {epoch}, Loss = {loss:.4f}")
                break

            prev_loss = loss

        plt.plot(loss_values)
        plt.title("Lasso Loss over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.show()
