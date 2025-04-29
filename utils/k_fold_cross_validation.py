import numpy as np

class KFoldCrossValidation:
    def __init__(self, k_folds=5):
        self.k_folds = k_folds

    def split(self, X, y):
        indices = np.random.permutation(len(X))
        X_shuffled, y_shuffled = X[indices], y[indices]
        fold_size = len(X) // self.k_folds
        folds = []

        for i in range(self.k_folds):
            start = i * fold_size
            end = start + fold_size if i != self.k_folds - 1 else len(X)
            X_val, y_val = X_shuffled[start:end], y_shuffled[start:end]
            X_train = np.concatenate([X_shuffled[:start], X_shuffled[end:]])
            y_train = np.concatenate([y_shuffled[:start], y_shuffled[end:]])
            folds.append((X_train, y_train, X_val, y_val))

        return folds

    def cross_validate(self, X, y, model):
        folds = self.split(X, y)
        accuracies = []

        for fold in folds:
            X_train, y_train, X_val, y_val = fold
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            accuracy = np.mean(y_pred == y_val)
            accuracies.append(accuracy)

        avg_accuracy = np.mean(accuracies)
        return avg_accuracy