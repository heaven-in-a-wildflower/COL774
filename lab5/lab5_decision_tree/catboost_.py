import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import time
import sys

class CatBoostModel:
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1, 
                 reg_lambda=1.0, random_state=42):
        self.model = CatBoostClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            reg_lambda=reg_lambda,
            random_state=random_state,
            verbose=1
        )
        self.model.set_params(allow_writing_files=False)

    def fit(self, X_train, y_train, X_test=None, y_test=None):
        print("\nStarting training...")
        print("Iter\tTrain Accuracy\tTest Accuracy")
        print("-" * 40)

        self.model.fit(
            X_train, y_train,
            eval_set=(X_test, y_test),
            use_best_model=True
        )

        # Calculate and print accuracies
        train_acc = self.model.score(X_train, y_train)
        test_acc = self.model.score(X_test, y_test) if X_test is not None else None
        
        if test_acc is not None:
            print(f"Final\t{train_acc:.4f}\t\t{test_acc:.4f}")
        else:
            print(f"Final\t{train_acc:.4f}")

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)
    
    def score(self, X, y):
        return self.model.score(X, y)

if __name__ == "__main__":
    t1 = time.time()
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    test_data = pd.read_csv(test_file)
    actual_labels = test_data.iloc[:,-1].values
    if 'target' in test_data.columns:
        test_data.drop('target', axis=1, inplace=True)

    # Load training data with header
    train_data = pd.read_csv(train_file)
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    X_test = test_data.values

    # Train model
    model = CatBoostModel(
        n_estimators=5000,
        max_depth=15,
        learning_rate=0.1,
        reg_lambda=0.01,
    )
    model.fit(X_train, y_train, X_test, actual_labels)
    
    # Make predictions
    y_pred = model.predict(X_train)
    accuracy = np.mean(y_pred == y_train)
    print(f"Training accuracy: {accuracy:.4f}")

    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == actual_labels)
    print(f"Test accuracy: {accuracy:.4f}")

    t2 = time.time()
    print(f"Time taken: {t2 - t1:.2f} seconds")