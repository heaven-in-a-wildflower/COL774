import numpy as np
import pandas as pd
import time
import sys
import lightgbm as lgb
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    t1 = time.time()
    
    # Parse command line arguments
    train_file = sys.argv[1]
    val_file = sys.argv[2]
    test_file = sys.argv[3]
    
    # Load data
    val_data = pd.read_csv(val_file)
    test_data = pd.read_csv(test_file)
    actual_labels = test_data.iloc[:,-1].values
    if 'target' in test_data.columns:
        test_data.drop('target', axis=1, inplace=True)

    # Load and combine training data
    train_data = pd.read_csv(train_file)
    train_data = pd.concat([train_data, val_data], axis=0, ignore_index=True)
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    X_test = test_data.values

    # Create dataset for LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)

    # Set parameters
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 255,  # Corresponds roughly to max_depth=8
        'learning_rate': 0.01,
        'feature_fraction': 0.8,  # Similar to colsample_bytree in XGBoost
        'bagging_fraction': 0.9,  # Similar to subsample in XGBoost
        'bagging_freq': 5,
        'verbose': 0,
        'num_threads': 4,
        'min_data_in_leaf': 2,  # Similar to min_samples_split
        'max_bin': 255,
        'seed': 42
    }

    print("\nStarting training...")
    print("Iter\tTrain Accuracy\tTest Accuracy")
    print("-" * 40)

    # Custom callback to print progress
    def callback(env):
        train_preds = env.model.predict(X_train, num_iteration=env.iteration + 1)
        train_acc = accuracy_score(y_train, (train_preds > 0.5).astype(int))
        
        test_preds = env.model.predict(X_test, num_iteration=env.iteration + 1)
        test_acc = accuracy_score(actual_labels, (test_preds > 0.5).astype(int))
        
        print(f"{env.iteration + 1:4d}\t{train_acc:.4f}\t\t{test_acc:.4f}")

    # Train model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=700,
        callbacks=[callback]
    )

    # Make predictions and calculate accuracies
    y_train_pred = (model.predict(X_train) > 0.5).astype(int)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"\nFinal Training accuracy: {train_accuracy:.4f}")

    y_test_pred = (model.predict(X_test) > 0.5).astype(int)
    test_accuracy = accuracy_score(actual_labels, y_test_pred)
    print(f"Final Test accuracy: {test_accuracy:.4f}")

    t2 = time.time()
    print(f"Time taken: {t2 - t1:.2f} seconds")

    # Optional: Feature importance analysis
    feature_importance = model.feature_importance(importance_type='gain')
    feature_names = [f'feature_{i}' for i in range(len(feature_importance))]
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10))