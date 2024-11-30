import numpy as np
import pandas as pd
import time
import sys
import numpy as np

class XGBDecisionTree:
    def __init__(self, max_depth=3, min_samples_split=3, learning_rate=0.1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.learning_rate = learning_rate
        self.tree = {}

    def _calc_leaf_value(self, grad, hess):
        """Newton-Raphson step"""
        return -np.sum(grad) / (np.sum(hess) + 1e-6)

    def _split_gain(self, grad, hess, left_grad, left_hess):
        """Calculate gain for a split"""
        def calc_term(g, h):
            return g * g / (h + 1e-6)
        gain = 0.5 * (calc_term(left_grad.sum(), left_hess.sum()) + 
                      calc_term(grad.sum() - left_grad.sum(), hess.sum() - left_hess.sum()) -
                      calc_term(grad.sum(), hess.sum()))
        return gain

    def _build_tree(self, X, grad, hess, depth=0, node_id=1):
        if depth >= self.max_depth or len(X) < self.min_samples_split:
            self.tree[node_id] = self._calc_leaf_value(grad, hess)
            return
        
        best_gain = 0
        best_feature = None
        best_threshold = None
        
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                left_grad = grad[left_mask]
                left_hess = hess[left_mask]
                
                if len(left_grad) == 0 or len(left_grad) == len(grad):
                    continue
                    
                gain = self._split_gain(grad, hess, left_grad, left_hess)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        if best_gain == 0:
            self.tree[node_id] = self._calc_leaf_value(grad, hess)
            return
            
        self.tree[node_id] = {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': node_id * 2,
            'right': node_id * 2 + 1
        }
        
        left_mask = X[:, best_feature] <= best_threshold
        self._build_tree(X[left_mask], grad[left_mask], hess[left_mask], 
                        depth + 1, node_id * 2)
        self._build_tree(X[~left_mask], grad[~left_mask], hess[~left_mask], 
                        depth + 1, node_id * 2 + 1)

    def _predict_one(self, x, node_id=1):
        if isinstance(self.tree[node_id], (int, float)):
            return self.tree[node_id]
        
        if x[self.tree[node_id]['feature']] <= self.tree[node_id]['threshold']:
            return self._predict_one(x, self.tree[node_id]['left'])
        return self._predict_one(x, self.tree[node_id]['right'])

    def fit(self, X, grad, hess):
        self._build_tree(X, grad, hess)
        
    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])

class XGBoost:
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1, 
                 min_samples_split=2, reg_lambda=1.0, reg_alpha=0.0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.trees = []
        self.base_score = 0.5

    def _grad(self, y_true, y_pred):
        """Gradient of logistic loss"""
        return y_pred - y_true

    def _hess(self, y_true, y_pred):
        """Hessian of logistic loss"""
        return y_pred * (1 - y_pred)

    def _logistic(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y, X_test=None, y_test=None):
        y_pred = np.full_like(y, self.base_score, dtype=float)
        
        print("\nStarting training...")
        print("Iter\tTrain Accuracy\tTest Accuracy")
        print("-" * 40)
        
        for i in range(self.n_estimators):
            grad = self._grad(y, self._logistic(y_pred))
            hess = self._hess(y, self._logistic(y_pred))
            
            tree = XGBDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                learning_rate=self.learning_rate
            )
            tree.fit(X, grad, hess)
            self.trees.append(tree)
            
            update = tree.predict(X)
            y_pred += self.learning_rate * update
            
            # Calculate and print accuracies
            train_acc = self.score(X, y)
            test_acc = self.score(X_test, y_test) if X_test is not None else None
            
            if test_acc is not None:
                print(f"{i+1:4d}\t{train_acc:.4f}\t\t{test_acc:.4f}")
            else:
                print(f"{i+1:4d}\t{train_acc:.4f}")


    def predict_proba(self, X):
        y_pred = np.full(X.shape[0], self.base_score, dtype=float)
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return self._logistic(y_pred)

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)

if __name__ == "__main__":
    t1 = time.time()
    train_file = sys.argv[1]
    val_file = sys.argv[2]
    val_data = pd.read_csv(val_file)
    test_file = sys.argv[3]
    test_data = pd.read_csv(test_file)
    actual_labels = test_data.iloc[:,-1].values
    if 'target' in test_data.columns:
        test_data.drop('target', axis=1, inplace=True)

    # Load training data with header
    train_data = pd.read_csv(train_file)
    train_data = pd.concat([train_data, val_data], axis=0, ignore_index=True)
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    X_test = test_data.values

    # Train model
    xgb = XGBoost(
        n_estimators=700,      # More trees (300->500)
        max_depth=8,           # Deeper trees (5->8)
        learning_rate=0.1,    # Smaller learning rate
        min_samples_split=2,   # Allow smaller splits (5->2)
    )
    xgb.fit(X_train, y_train, X_test, actual_labels)
    
    # Make predictions
    y_pred = xgb.predict(X_train)
    accuracy = np.mean(y_pred == y_train)
    print(f"Training accuracy: {accuracy:.4f}")

    y_pred = xgb.predict(X_test)
    accuracy = np.mean(y_pred == actual_labels)
    print(f"Test accuracy: {accuracy:.4f}")


    t2 = time.time()
    print(f"Time taken: {t2 - t1:.2f} seconds")

    