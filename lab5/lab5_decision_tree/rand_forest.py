import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import json
import sys
import math
import time

class RandomForestTree:
    def __init__(self, max_depth, min_samples_split, min_samples_leaf, ccp_alpha):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.ccp_alpha = ccp_alpha
        self.tree = {}

    def fit(self, X, y):
        print(f"Training tree with {len(X)} samples")
        self.tree = self._build_tree(X, y, depth=0, node_id=1)

    def _get_majority_class(self, y):
        counts = np.bincount(y.astype(int))
        if len(counts) == 0:
            return -1
        return np.argmax(counts)

    def _build_tree(self, X, y, depth, node_id):
        # Early stopping conditions
        if depth >= self.max_depth or len(X) < self.min_samples_split or len(X) < self.min_samples_leaf:
            return {'type': 'leaf', 'class': self._get_majority_class(y)}

        # Check if all samples belong to the same class
        unique_classes = np.unique(y)
        if len(unique_classes) == 1:
            return {'type': 'leaf', 'class': unique_classes[0]}

        try:
            # Fit logistic regression
            lr = LogisticRegression(fit_intercept=False)
            lr.fit(X, y)
            weights = lr.coef_[0]

            predictions = np.dot(X, weights)
            sorted_indices = np.argsort(predictions)
            sorted_predictions = predictions[sorted_indices]
            sorted_y = y[sorted_indices]

            best_gini = float('inf')
            best_threshold = None

            # Find best split
            for i in range(1, len(sorted_predictions)):
                threshold = (sorted_predictions[i-1] + sorted_predictions[i]) / 2
                left_mask = predictions <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                gini = self._gini_impurity(sorted_y[sorted_predictions <= threshold], 
                                           sorted_y[sorted_predictions > threshold])
                if gini < best_gini:
                    best_gini = gini
                    best_threshold = threshold

            if best_threshold is None:
                return {'type': 'leaf', 'class': self._get_majority_class(y)}

            left_mask = predictions <= best_threshold
            right_mask = ~left_mask

            node = {
                'type': 'node',
                'weights': weights,
                'threshold': best_threshold
            }

            node['left'] = self._build_tree(X[left_mask], y[left_mask], depth + 1, node_id * 2)
            node['right'] = self._build_tree(X[right_mask], y[right_mask], depth + 1, node_id * 2 + 1)

            return node

        except Exception as e:
            print(f"Error in building node: {str(e)}")
            return {'type': 'leaf', 'class': self._get_majority_class(y)}

    def _gini_impurity(self, left_y, right_y):
        def gini(y):
            m = len(y)
            return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in np.unique(y))

        m = len(left_y) + len(right_y)
        return (len(left_y) / m) * gini(left_y) + (len(right_y) / m) * gini(right_y)

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

    def _predict_one(self, x, node):
        if node['type'] == 'leaf':
            return node['class']
        if np.dot(x, node['weights']) <= node['threshold']:
            return self._predict_one(x, node['left'])
        return self._predict_one(x, node['right'])

class ObliqueRandomForest:
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=2, min_samples_leaf=1, ccp_alpha=0.0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.ccp_alpha = ccp_alpha
        self.trees = []

    def fit(self, X, y, X_test=None, y_test=None):
        print(f"Training Random Forest with {self.n_estimators} trees...")
        
        for i in range(self.n_estimators):
            print(f"Training tree {i+1}/{self.n_estimators}")
            tree = RandomForestTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf, ccp_alpha=self.ccp_alpha)
            
            # Bootstrap sampling for each tree
            bootstrap_indices = np.random.choice(len(X), size=len(X), replace=True)
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
            
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
            
            # Calculate and print accuracies
            train_acc = self.score(X, y)
            test_acc = self.score(X_test, y_test) if X_test is not None else None
            
            if test_acc is not None:
                print(f"Tree {i+1}/{self.n_estimators} - Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")
            else:
                print(f"Tree {i+1}/{self.n_estimators} - Train Accuracy: {train_acc:.4f}")
        
        print("Random Forest training completed.")

    def predict(self, X):
        # Get predictions from all trees
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # Return majority vote
        return np.array([np.bincount(pred).argmax() for pred in predictions.T])

    def score(self, X, y):
        if X is None or y is None:
            return None
        return np.mean(self.predict(X) == y)

    def save_model(self, filename):
        # Save model parameters as JSON
        model_params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'ccp_alpha': self.ccp_alpha
        }
        
        with open(filename, 'w') as f:
            json.dump(model_params, f)

if __name__ == "__main__":
    command = sys.argv[1]
    t1 = time.time()

    if command == 'train':
        mode = sys.argv[2]  # 'pruned' or 'unpruned'
        train_file = sys.argv[3]
        
        # Load training data
        train_data = pd.read_csv(train_file)
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values
        
        # Initialize random forest
        rf = ObliqueRandomForest(
            n_estimators=100,
            max_depth=int(sys.argv[4] if mode == 'unpruned' else sys.argv[5]),
            min_samples_split=5,
            min_samples_leaf=2,
            ccp_alpha=0.01  # Cost complexity pruning alpha
        )
        
        if mode == 'unpruned':
            rf.fit(X_train, y_train)
            
            # Calculate and save training predictions
            train_preds = rf.predict(X_train)
            pd.DataFrame(train_preds).to_csv("train_predictions.csv", header=False, index=False)
            accuracy = np.mean(train_preds == y_train)
            print(f"Training accuracy without pruning: {accuracy:.4f}")
            
        else:  # pruned mode
            val_file = sys.argv[4]
            
            # Load validation data
            val_data = pd.read_csv(val_file)
            X_val = val_data.iloc[:, :-1].values
            y_val = val_data.iloc[:, -1].values
            
            # Combine train and validation data
            combined_data = pd.concat([train_data, val_data], axis=0, ignore_index=True)
            X_combined = combined_data.iloc[:, :-1].values
            y_combined = combined_data.iloc[:, -1].values
            
            rf.fit(X_combined, y_combined, X_test=X_val, y_test=y_val)
            
            # Calculate and save training predictions
            train_preds = rf.predict(X_combined)
            pd.DataFrame(train_preds).to_csv("train_predictions.csv", header=False, index=False)
            train_accuracy = np.mean(train_preds == y_combined)
            print(f"Training accuracy with pruning: {train_accuracy:.4f}")
            
            # Calculate validation accuracy
            val_preds = rf.predict(X_val)
            val_accuracy = np.mean(val_preds == y_val)
            print(f"Validation accuracy: {val_accuracy:.4f}")
    
    elif command == 'test':
        train_file = sys.argv[2]
        val_file = sys.argv[3]
        test_file = sys.argv[4]
        max_depth = int(sys.argv[5])
        prediction_file = sys.argv[6]
        
        # Load datasets
        train_data = pd.read_csv(train_file)
        val_data = pd.read_csv(val_file)
        test_data = pd.read_csv(test_file)
        
        # Combine train and validation data
        combined_data = pd.concat([train_data, val_data], axis=0, ignore_index=True)
        X_train = combined_data.iloc[:, :-1].values
        y_train = combined_data.iloc[:, -1].values
        X_test = test_data.values
        
        if 'target' in test_data.columns:
            actual_labels = test_data['target'].values
            test_data.drop('target', axis=1, inplace=True)
        X_test = test_data.values
        
        # Train random forest
        rf = ObliqueRandomForest(
            n_estimators=100,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            ccp_alpha=0.05  # Cost complexity pruning alpha
        )
        rf.fit(X_train, y_train, X_test=X_test, y_test=actual_labels)
        
        # Generate and save predictions
        train_preds = rf.predict(X_train)
        test_preds = rf.predict(X_test)
        
        pd.DataFrame(test_preds).to_csv(prediction_file, header=False, index=False)
        
        # Calculate and print accuracies
        train_accuracy = np.mean(train_preds == y_train)
        print(f"Training accuracy: {train_accuracy:.4f}")
        
        if 'actual_labels' in locals():
            test_accuracy = np.mean(test_preds == actual_labels)
            print(f"Test accuracy: {test_accuracy:.4f}")
    
    t2 = time.time()
    print(f"Total execution time: {t2 - t1:.2f} seconds")