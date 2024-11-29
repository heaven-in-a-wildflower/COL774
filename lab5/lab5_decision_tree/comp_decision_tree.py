import numpy as np
import pandas as pd
import sys
from sklearn.linear_model import LogisticRegression

class ObliqueDecisionTree:
    def __init__(self, max_depth, min_samples_split=3):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.nodes = {}  # Node store dictionary

    def fit(self, X, y, mode='unpruned', X_val=None, y_val=None):
        print(f"Starting to build the tree in {mode} mode...")
        self.root = self._build_tree(X, y, depth=0, node_id=1)
        
        if mode == 'pruned' and X_val is not None and y_val is not None:
            print("Pruning the tree using validation data...")
            self._prune_tree(self.root, X_val, y_val)

    def _get_majority_class(self, y):
        counts = np.bincount(y.astype(int))
        if len(counts) == 0:
            return -1
        if len(counts) == 1:
            return 0
        return 0 if counts[0] >= counts[1] else 1

    def _build_tree(self, X, y, depth, node_id):
        if depth >= self.max_depth or len(X) < self.min_samples_split or len(np.unique(y)) == 1:
            majority_class = self._get_majority_class(y)
            self.nodes[node_id] = {'type': 'leaf', 'class_label': majority_class}
            return node_id

        #weights = logistic_regression(X, y)
        lr = LogisticRegression(fit_intercept=False)
        lr.fit(X, y)
        weights = lr.coef_[0]
        predictions = np.dot(X, weights)
        sorted_indices = np.argsort(predictions)
        sorted_predictions = predictions[sorted_indices]
        sorted_y = y[sorted_indices]

        best_gini = float('inf')
        best_threshold = None
        for i in range(1, len(sorted_predictions)):
            threshold = (sorted_predictions[i-1] + sorted_predictions[i]) / 2
            left_mask = sorted_predictions <= threshold
            right_mask = sorted_predictions > threshold
            gini = self._gini_impurity(sorted_y[left_mask], sorted_y[right_mask])
            if gini < best_gini:
                best_gini = gini
                best_threshold = threshold

        left_mask = predictions <= best_threshold
        right_mask = predictions > best_threshold

        self.nodes[node_id] = {
            'type': 'node',
            'weights': weights,
            'threshold': best_threshold,
            'majority_class': self._get_majority_class(y)
        }

        if len(X[left_mask]) == 0 or len(X[right_mask]) == 0:
            self.nodes[node_id]['type'] = 'leaf'
            self.nodes[node_id]['class_label'] = self._get_majority_class(y)
            return node_id

        left_id = self._build_tree(X[left_mask], y[left_mask], depth + 1, node_id * 2)
        right_id = self._build_tree(X[right_mask], y[right_mask], depth + 1, node_id * 2 + 1)
        
        self.nodes[node_id]['left'] = left_id
        self.nodes[node_id]['right'] = right_id
        
        return node_id

    def _predict_one(self, x, node_id):
        node = self.nodes[node_id]
        if node['type'] == 'leaf':
            return node['class_label']
        if np.dot(x, node['weights']) <= node['threshold']:
            return self._predict_one(x, node['left'])
        return self._predict_one(x, node['right'])

    def _prune_tree(self, node_id, X_val, y_val):
        node = self.nodes[node_id]
        if node['type'] == 'leaf':
            return

        if 'left' in node:
            self._prune_tree(node['left'], X_val, y_val)
        if 'right' in node:
            self._prune_tree(node['right'], X_val, y_val)

        accuracy_before = self._evaluate_accuracy(X_val, y_val)
        
        original_node = node.copy()
        node.clear()
        node.update({
            'type': 'leaf',
            'class_label': original_node['majority_class']
        })
        
        accuracy_after = self._evaluate_accuracy(X_val, y_val)
        
        if accuracy_before > accuracy_after:
            node.clear()
            node.update(original_node)

    def _evaluate_accuracy(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def _gini_impurity(self, left_y, right_y):
        def gini(y):
            m = len(y)
            return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in np.unique(y))
        m = len(left_y) + len(right_y)
        return (len(left_y) / m) * gini(left_y) + (len(right_y) / m) * gini(right_y)

    def predict(self, X):
        return np.array([self._predict_one(x, 1) for x in X])

    def _predict_one(self, x, node_id):
        node = self.nodes[node_id]
        if node['type'] == 'leaf':
            return node['class_label']
        if np.dot(x, node['weights']) <= node['threshold']:
            return self._predict_one(x, node['left'])
        return self._predict_one(x, node['right'])

    def save_predictions(self, X, filename):
        predictions = self.predict(X)
        np.savetxt(filename, predictions, fmt='%d')

    def save_weights(self, filename):
        with open(filename, 'w') as f:
            self._write_weights_recursive(self.root, f)

    def _write_weights_recursive(self, node_id, f):
        node = self.nodes[node_id]
        if node['type'] == 'node':
            weights_str = ','.join(map(str, node['weights']))
            f.write(f"{node_id},{weights_str},{node['threshold']}\n")
            self._write_weights_recursive(node['left'], f)
            self._write_weights_recursive(node['right'], f)

if __name__ == "__main__":
    command = sys.argv[1]
    if command == 'train':
        mode = sys.argv[2]
        train_file = sys.argv[3]
        
        train_data = pd.read_csv(train_file)
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values
        
        tree = ObliqueDecisionTree(max_depth=int(sys.argv[4] if mode == 'unpruned' else sys.argv[5]))
        
        if mode == 'unpruned':
            weights_file = sys.argv[5]
            tree.fit(X_train, y_train, mode='unpruned')
            tree.save_predictions(X_train,"train_predictions.csv")
            predictions = pd.read_csv("train_predictions.csv",header=None)
            predictions = predictions.iloc[:, -1].values
            accuracy = np.mean(predictions == y_train)
            print("Accuracy of the model on train data without pruning is: ",accuracy)
        else:
            val_file = sys.argv[4]
            weights_file = sys.argv[6]
            
            val_data = pd.read_csv(val_file)
            X_val = val_data.iloc[:, :-1].values
            y_val = val_data.iloc[:, -1].values
            
            tree.fit(X_train, y_train, mode='pruned', X_val=X_val, y_val=y_val)
        
        tree.save_weights(weights_file)
        
    elif command == 'test':
        train_file = sys.argv[2]
        val_file = sys.argv[3]
        test_file = sys.argv[4]
        prediction_file = sys.argv[5]
        weights_file = sys.argv[6]
        max_depth = 9
        
        # Load all datasets (test file has no target)
        train_data = pd.read_csv(train_file)
        val_data = pd.read_csv(val_file)
        test_data = pd.read_csv(test_file)
        # actual_labels = test_data.iloc[:,-1].values
        if 'target' in test_data.columns:
            test_data.drop('target', axis=1, inplace=True)
        print(test_data.shape[1])
        X_train = train_data.iloc[:, :-1].values
        print(X_train.shape[1])
        y_train = train_data.iloc[:, -1].values
        X_val = val_data.iloc[:, :-1].values
        print(X_val.shape[1])   
        y_val = val_data.iloc[:, -1].values
        X_test = test_data.values
        
        # Train pruned tree and make predictions
        tree = ObliqueDecisionTree(max_depth=max_depth)
        tree.fit(X_train, y_train, mode='pruned', X_val=X_val, y_val=y_val)
        
        # tree.save_predictions(X_train,"train_predictions.csv")
        # predictions = pd.read_csv("train_predictions.csv",header=None)
        # predictions = predictions.iloc[:, -1].values
        # accuracy = np.mean(predictions == y_train)
        # print("Accuracy of the model on train data is: ",accuracy)

        # Save predictions and calculate accuracy on the validation data
        # tree.save_predictions(X_val,"val_predictions.csv")
        # predictions = pd.read_csv("val_predictions.csv",header=None)
        # predictions = predictions.iloc[:, -1].values
        # accuracy = np.mean(predictions == y_val)
        # print("Accuracy of the model on validation data is: ",accuracy)

        tree.save_predictions(X_test, prediction_file)
        # print(actual_labels)
        # predictions = pd.read_csv(prediction_file,header=None)
        # predictions = predictions.iloc[:, -1].values
        # print(predictions)
        # accuracy = np.mean(predictions == actual_labels)
        # print("Accuracy of the model is: ",accuracy)

        tree.save_weights(weights_file)
