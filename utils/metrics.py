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
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.cm = None
    
    def compute(self, y_true, y_pred):
        """Compute the confusion matrix for multi-class classification."""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        
        # Initialize the confusion matrix with zeros
        self.cm = np.zeros((self.num_classes, self.num_classes), dtype=int)
        
        # Loop through each prediction and actual value to update the confusion matrix
        for true, pred in zip(y_true, y_pred):
            self.cm[true, pred] += 1
        
        return self.cm

    def plot(self):
        """Plot the confusion matrix using a heatmap."""
        if self.cm is None:
            raise ValueError("Confusion matrix is not computed. Please compute it first.")
        
        # Generate a list of class labels based on the number of classes
        labels = [f'Class {i}' for i in range(self.num_classes)]
        
        # Plot the confusion matrix
        plt.figure(figsize=(6, 6))
        sns.heatmap(self.cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

