import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class PrecisionScore:
    def __init__(self):
        self.precision = 0
    
    def compute(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        
        if TP + FP == 0:
            return 0.0
        self.precision = TP / (TP + FP)
        return self.precision

class RecallScore:
    def __init__(self):
        self.recall = 0
    
    def compute(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        TP = np.sum((y_true == 1) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        
        if TP + FN == 0:
            return 0.0
        self.recall = TP / (TP + FN)
        return self.recall

class F1Score:
    def __init__(self):
        self.f1 = 0
    
    def compute(self, precision, recall):
        if precision + recall == 0:
            return 0.0
        self.f1 = 2 * (precision * recall) / (precision + recall)
        return self.f1

class ConfusionMatrix:
    def __init__(self):
        self.cm = None
    
    def compute(self, y_true, y_pred):
        """Compute the confusion matrix."""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        
        # Initialize the confusion matrix
        TP = np.sum((y_true == 1) & (y_pred == 1))  # True Positives
        TN = np.sum((y_true == 0) & (y_pred == 0))  # True Negatives
        FP = np.sum((y_true == 0) & (y_pred == 1))  # False Positives
        FN = np.sum((y_true == 1) & (y_pred == 0))  # False Negatives
        
        # Store the confusion matrix
        self.cm = np.array([[TN, FP], [FN, TP]])
        
        return self.cm

    def plot(self):
        """Plot the confusion matrix."""
        if self.cm is None:
            raise ValueError("Confusion matrix is not computed. Please compute it first.")
        
        plt.figure(figsize=(6, 6))
        sns.heatmap(self.cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
