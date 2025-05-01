from collections import defaultdict, Counter
import math

class NaiveBayes:
    def __init__(self):
        # Initialize the counters and vocabulary
        self.class_word_counts = {
            'ham': Counter(),
            'spam': Counter()
        }
        self.class_counts = {
            'ham': 0,
            'spam': 0
        } 
        self.vocabulary = set()

    def fit(self, X, y):
        # Count word frequencies per class
        for label, tokens in zip(y, X):
            self.class_counts[label] += 1
            self.class_word_counts[label].update(tokens)

        # Create vocabulary (all unique words across both classes)
        self.vocabulary = set(self.class_word_counts['ham'].keys()) | set(self.class_word_counts['spam'].keys())

    def get_class_counts(self):
        return self.class_counts

    def get_class_word_counts(self):
        return self.class_word_counts

    def get_vocabulary(self):
        return self.vocabulary
    
    def predict(self, X):
        predictions = []

        total_docs = sum(self.class_counts.values())
        vocab_size = len(self.vocabulary)

        # Total number of words in each class
        total_words = {
            label: sum(self.class_word_counts[label].values())
            for label in self.class_counts
        }

        for tokens in X:
            class_scores = {}

            for label in self.class_counts:
                # Start with log prior
                log_prob = math.log(self.class_counts[label] / total_docs)

                for word in tokens:
                    # Count of word in this class
                    word_count = self.class_word_counts[label][word]
                    
                    # Apply Laplace smoothing
                    word_prob = (word_count + 1) / (total_words[label] + vocab_size)
                    log_prob += math.log(word_prob)

                class_scores[label] = log_prob

            # Choose the class with higher score
            predicted_class = max(class_scores, key=class_scores.get)
            predictions.append(predicted_class)

        return predictions
    
    def score(self, X, y_true):
        y_pred = self.predict(X)
        correct = sum(pred == true for pred, true in zip(y_pred, y_true))
        return correct / len(y_true)
