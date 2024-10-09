import pandas as pd
import numpy as np
import sys
from scipy.stats import multinomial
from scipy.optimize import minimize
from sklearn import preprocessing

def load_data(train_file):
    train_data = pd.read_csv(train_file)
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    return X_train, y_train

def load_params(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        strategy = int(lines[0].strip())
        
        if strategy == 1:
            rate_params = [np.float64(lines[1].strip())]
        elif strategy == 2:
            rate_params = list(map(np.float64, lines[1].strip().split(',')))
        elif strategy == 3:
            rate_params = [np.float64(lines[1].strip())]
            print(rate_params)

        epochs = int(lines[2].strip())
        batch_size = int(lines[3].strip())
    
    return strategy, rate_params, batch_size, epochs

def loss(X, y, w, freq):
    logits = X @ w
    max_logits = np.max(logits, axis=1, keepdims=True)
    logits = logits - max_logits
    
    exp_logits = np.exp(logits)
    sum_exp_logits = np.sum(exp_logits, axis=1, keepdims=True)

    log_p = logits - np.log(sum_exp_logits)
    l = 0
    true_class_indices = np.argmax(y, axis=1)
    l = -np.sum(np.sum(y * log_p, axis=1) / freq[true_class_indices])
    
    return l / (2 * X.shape[0])

def compute_grad(X, y, w, freq):
    n = X.shape[0]  
    m = X.shape[1]  
    k = w.shape[1]  

    
    logits = X @ w
    max_logits = np.max(logits, axis=1, keepdims=True)
    logits = logits - max_logits

    exp_logits = np.exp(logits)
    sum_exp_logits = np.sum(exp_logits, axis=1, keepdims=True)
    p = exp_logits / sum_exp_logits
    error = p - y
    true_class_indices = np.argmax(y, axis=1) 
    freq_adjustment = np.array([freq[idx] for idx in true_class_indices])
    adjusted_X = X / freq_adjustment[:, np.newaxis]
    grad = (adjusted_X.T @ error) / (2 * n)

    return grad

def loss_newton(w, X, y, freq, len_classes):
    w = w.reshape((X.shape[1], len_classes))
    logits = X @ w
    max_logits = np.max(logits, axis=1, keepdims=True)
    logits = logits - max_logits
    
    exp_logits = np.exp(logits)
    sum_exp_logits = np.sum(exp_logits, axis=1, keepdims=True)
    log_p = logits - np.log(sum_exp_logits)
    l = 0
    true_class_indices = np.argmax(y, axis=1)
    l = -np.sum(np.sum(y * log_p, axis=1) / freq[true_class_indices])
    return l / (2 * X.shape[0])

def compute_grad_newton(w, X, y, freq, len_classes):
    w = w.reshape((X.shape[1],len_classes))
    logits = X @ w
    max_logits = np.max(logits, axis=1, keepdims=True)
    logits = logits - max_logits
    exp_logits = np.exp(logits)
    sum_exp_logits = np.sum(exp_logits, axis=1, keepdims=True)
    p = exp_logits / sum_exp_logits
    error = p - y
    true_class_indices = np.argmax(y, axis=1)
    freq_adjustment = np.array([freq[idx] for idx in true_class_indices])
    adjusted_X = X / freq_adjustment[:, np.newaxis]
    grad = (adjusted_X.T @ error) / (2 * X.shape[0])
    return grad.flatten()

def compute_hessian(w,X,y,freq,len_classes):
    n = X.shape[0]  
    m = X.shape[1]  

    w = w.reshape((X.shape[1],len_classes))
    k = len_classes

    logits = X @ w
    max_logits = np.max(logits, axis=1, keepdims=True)
    logits = logits - max_logits
    exp_logits = np.exp(logits)
    sum_exp_logits = np.sum(exp_logits, axis=1, keepdims=True)
    p = exp_logits / sum_exp_logits

    error = p - y

    J = np.zeros((n * k, n * k))
    for i in range(n):
        for j in range(k):
            for l in range(k):
                if j == l:
                    J[i * k + j, i * k + j] = p[i, j] * (1 - p[i, j])
                else:
                    J[i * k + j, i * k + l] = -p[i, j] * p[i, l]

    true_class_indices = np.argmax(y, axis=1)  
    freq_adjustment = np.array([freq[idx] for idx in true_class_indices])
    adjusted_X = X / freq_adjustment[:, np.newaxis]

    # Reshape X to match Jacobian dimensions
    adjusted_X_expanded = np.zeros((n * k, m))
    for i in range(n):
        adjusted_X_expanded[i * k:(i + 1) * k, :] = adjusted_X[i, :]

    # Compute the Hessian
    H = (adjusted_X_expanded.T @ J @ adjusted_X_expanded) / (2 * n)

    return H.flatten()

def hessian_vector_product(w, v, X, y, freq,len_classes):
    w = w.reshape((X.shape[1],len_classes))
    # Reshape vector v to match the dimensions of w
    m, k = w.shape
    v = v.reshape((m, k))

    # Compute logits
    logits = X @ w
    max_logits = np.max(logits, axis=1, keepdims=True)
    logits = logits - max_logits
    exp_logits = np.exp(logits)
    sum_exp_logits = np.sum(exp_logits, axis=1, keepdims=True)
    p = exp_logits / sum_exp_logits

    # Adjust X by dividing each row by the corresponding frequency of the true class
    true_class_indices = np.argmax(y, axis=1)
    freq_adjustment = np.array([freq[idx] for idx in true_class_indices])
    adjusted_X = X / freq_adjustment[:, np.newaxis]

    # Compute the Jacobian-vector product (J * v)
    Jv = np.zeros_like(w)
    for i in range(len(y)):
        pi = p[i]
        yi = y[i]
        x_adj = adjusted_X[i]
        for j in range(len(pi)):
            for l in range(len(pi)):
                if j == l:
                    Jv[:, j] += (pi[j] * (1 - pi[j]) * (x_adj @ v[:, j])) * x_adj
                else:
                    Jv[:, j] -= (pi[j] * pi[l] * (x_adj @ v[:, l])) * x_adj

    Hvp = Jv / (2 * len(y))
    return Hvp.flatten()

def ternary_search(X, y, w, g, eta_0, freq, max_iter=20):
    eta_l = 0
    eta_h = eta_0
    
    while loss(X, y, w, freq) > loss(X, y, w - eta_h * g, freq):
        eta_h *= 2

    for _ in range(max_iter):
        eta_1 = (2 * eta_l + eta_h) / 3
        eta_2 = (eta_l + 2 * eta_h) / 3

        if loss(X, y, w - eta_1 * g, freq) > loss(X, y, w - eta_2 * g, freq):
            eta_l = eta_1
        else:
            eta_h = eta_2

    eta = (eta_l + eta_h) / 2
    return eta

def gradient_descent(X, y, w, freq, rate_params, length, epochs, strategy):
    eta_0 = rate_params[0]
    k = rate_params[1] if strategy == 2 else 0  # k = 0 for strategy 1
    n = X.shape[0]
    for epoch in range(epochs):
        for i in range(0, n, length):
            X_batch = X[i:i+length]
            y_batch = y[i:i+length]

            if strategy == 3:
                g = compute_grad(X_batch, y_batch, w, freq)
                #print("gradient is", g)
                eta = ternary_search(X_batch, y_batch, w, g, rate_params[0], freq)
                #print("eta is ", eta)
                rate = eta
            elif strategy == 2:
                rate = eta_0 / (1 + k * (1 + epoch))
            else:
                rate = eta_0
            batch_loss = 0
            batch_loss = loss(X_batch, y_batch, w, freq)
            log_message = f"Epoch Number: {epoch + 1}, Batch Number: {i // length + 1}, Batch Loss(before updating weights): {batch_loss}"
            if(i // length + 1 ==1):
                print(log_message)
            if strategy==3:
                w -= rate * g
            else:
                grad = compute_grad(X_batch, y_batch, w, freq)
                w -= rate * grad
    return w

def part_a(train_file, params_file, weights_file):
    
    X_train, y_train = load_data(train_file)
    
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    
    classes = np.unique(y_train)
    
    y_onehot = np.zeros((y_train.shape[0], len(classes)))
    
    for idx, val in enumerate(classes):
        y_onehot[y_train == val, idx] = 1
    
    freq = np.array([np.sum(y_train == x) for x in classes])

    w = np.zeros((X_train.shape[1], len(classes)), dtype=np.float64)
    
    strategy, rate_params, batch_size, epochs = load_params(params_file)
    
    print(strategy, rate_params, batch_size, epochs)
    
    w = gradient_descent(X_train, y_onehot, w, freq, rate_params, batch_size, epochs, strategy)
    print(np.shape(w),np.shape(X_train),np.shape(y_train),np.shape(y_onehot))
    
    np.savetxt(weights_file, w.flatten().astype(np.float64), fmt='%.16e')

def part_b(train_file, test_file, weights_file, prediction_file):
    X_train, y_train = load_data(train_file)
    
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_train = np.insert(X_train,0,np.ones(X_train.shape[0]),axis=1) 
    # Fit the scaler on therate_params feature columns and transform them
    # X_train = scaler.fit_transform(X_train)

    # Reattach the intercept column (first column)
    # X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))

    classes = np.unique(y_train)
    
    y_onehot = np.zeros((y_train.shape[0], len(classes)))
    
    for idx, val in enumerate(classes):
        y_onehot[y_train == val, idx] = 1
    
    freq = np.array([np.sum(y_train == x) for x in classes])

    w = np.zeros((X_train.shape[1],len(classes)), dtype=np.float64)

    # ****************** Customisable portion for part-b ********************* 

    strategy, rate_params, batch_size, epochs = 3,[np.float64('1e-9')],np.shape(X_train)[0],15
    
    # print(strategy, rate_params, batch_size, epochs)
    print("Hello")
    w = gradient_descent(X_train, y_onehot, w, freq, rate_params, batch_size, epochs, strategy)
    
    # result = minimize(loss_newton, w.flatten(), args=(X_train, y_onehot, freq, len(classes)),
    #               method='Powell', options={'disp': True, 'maxiter': 10})

    # result = minimize(loss_newton, w.flatten(), args=(X_train, y_onehot, freq, len(classes)),
    #               method='trust-krylov', jac=compute_grad_newton, hessp=hessian_vector_product,
    #               options={'disp': True, 'maxiter': 10,'gtol': 1e-12})

    # Extract the optimal weights
    # w = result.x.reshape((X_train.shape[1], len(classes)))    

    print(np.shape(w),np.shape(X_train),np.shape(y_train),np.shape(y_onehot))

    # ************************************************************************

    # Predict probabilities
    # Load test data
    test_data = pd.read_csv(test_file)
    X_test = test_data.values

    X_test = scaler.transform(X_test)
    X_test = np.insert(X_test,0,np.ones(X_test.shape[0]),axis=1)
    # X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

    # Compute softmax probabilities
    z = np.dot(X_test, w)
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Subtract max for numerical stability
    softmax_probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)

    # Uncomment later
    np.savetxt(weights_file, w.flatten().astype(np.float64), fmt='%.16e')

    # Save softmax probabilities to a CSV file without a header
    np.savetxt(prediction_file, softmax_probs, delimiter=",")

if __name__ == "__main__":
    if sys.argv[1] == 'a':
        part_a(sys.argv[2], sys.argv[3], sys.argv[4])
    if sys.argv[1] == 'b':
        part_b(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])