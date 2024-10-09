import numpy as np
import pandas as pd
import sys

def ridge_regression(X, y, l):
    I = np.eye(X.shape[1])
    w = np.linalg.inv(X.T @ X + l * I) @ X.T @ y
    return w

def cross_validation(X, y, lambdas):
    min_error = float('inf')
    l_min = None
    size = X.shape[0] // 10
    for l in lambdas:
        start = 0
        end = size
        curr_error = 0
        for i in range(10):
            X_test = X[start:end]
            y_test = y[start:end]
            X_train = np.concatenate((X[:start], X[end:]), axis=0)
            y_train = np.concatenate((y[:start], y[end:]), axis=0)

            w = ridge_regression(X_train, y_train, l)
            y_pred = X_test @ w
            curr_error += np.mean((y_test - y_pred) ** 2)
            start += size
            end += size
        print(l, curr_error)
        if curr_error < min_error:
            min_error = curr_error
            l_min = l
    return l_min

# Read command-line arguments
part = sys.argv[1]
train_file = sys.argv[2]
test_file = sys.argv[3]
weights_file = sys.argv[4]
predictions_file = sys.argv[5]
modelweights_file = sys.argv[6]

# Load data
train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.values

# Add bias term
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

if part == 'a':
    # Weighted Linear Regression
    input_weights = np.loadtxt(weights_file)
    sqrt_weights = np.sqrt(input_weights)
    new_X_train = X_train * sqrt_weights[:, np.newaxis]
    new_y_train = y_train * sqrt_weights

    params = np.linalg.inv(new_X_train.T @ new_X_train) @ new_X_train.T @ new_y_train
    pred = X_test @ params

    np.savetxt(predictions_file, pred)
    np.savetxt(modelweights_file, params)

elif part == 'b':
    # Ridge Regression with Cross-Validation
    X_train = X_train[:-4]
    y_train = y_train[:-4]

    # Load lambda values
    with open(weights_file, 'r') as f:
        lambdas = [float(line.strip()) for line in f]

    best_lambda = cross_validation(X_train, y_train, lambdas)
    params = ridge_regression(X_train, y_train, best_lambda)
    pred = X_test @ params

    np.savetxt(predictions_file, pred)
    np.savetxt(modelweights_file, params)
    with open('bestlambda.txt', 'w') as f:
        f.write(str(best_lambda))
