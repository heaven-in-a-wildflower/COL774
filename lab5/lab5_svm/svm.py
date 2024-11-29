import cvxpy as cp
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler

def train_svm(X, y, C=1):
    """
    Train SVM classifier using CVXPY with L1 norm of weights
    
    Args:
        X: Feature matrix
        y: Labels (0/1)
        C: Regularization parameter (default=1)
    
    Returns:
        weights: Optimal weights
        bias: Optimal bias
        support_vectors: Indices of support vectors
        xi_values: Values of the slack variables
    """
    n_samples, n_features = X.shape
    
    # Define variables
    w = cp.Variable(n_features)
    b = cp.Variable()
    xi = cp.Variable(n_samples)
    
    # Define objective function (using L1 norm)
    objective = cp.Minimize(0.5 * cp.norm1(w) + C * cp.sum(xi))
    
    # Define constraints using original 0/1 labels
    constraints = []
    for i in range(n_samples):
        if y[i] == 1:
            constraints.append(X[i] @ w + b >= 1 - xi[i])
        else:
            constraints.append(X[i] @ w + b <= -1 + xi[i])
    constraints.append(xi >= 0)
    
    # Solve the optimization problem
    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    if problem.status != cp.OPTIMAL:
        return None, None, [], None
    
    # Get the weights, bias, and slack variables
    weights = w.value
    bias = b.value
    xi_values = xi.value
    
    # Find support vectors (points close to the margin)
    margins = np.abs(X @ weights + bias)
    sv_tolerance = 1e-4
    support_vectors = np.where(np.abs(margins - 1) <= sv_tolerance)[0].tolist()
    
    return weights, bias, support_vectors, xi_values

def is_linearly_separable(xi_values, tolerance=1e-4):
    """
    Check if a dataset is linearly separable based on the values of the slack variables (xi).
    
    Parameters:
    xi_values (array-like): Array of slack variable values from the SVM optimization.
    tolerance (float): A small threshold value to check if slack values are close to zero.
    
    Returns:
    bool: True if the dataset is linearly separable, False otherwise.
    """
    # Check if all slack variables are within the tolerance range of 0
    separable = np.all(xi_values <= tolerance)
    if separable==True:
        print('Separable')
    else:
        print('Not separable')
    return separable

def process_dataset(filename):
    """
    Process a dataset and create corresponding weight and sv json files
    """
    # Read the dataset
    data = pd.read_csv(filename)
    
    # Extract features and target
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train SVM
    weights, bias, support_vectors, xi_values = train_svm(X, y)
    
    if weights is None:
        # If optimization failed, return default values
        weights = np.zeros(X.shape[1])
        bias = 0
        support_vectors = []
        separable = 0
    else:
        # Check if dataset is linearly separable based on xi values
        separable = 1 if is_linearly_separable(xi_values) else 0
    
    # Create weight json
    weight_dict = {
        "weights": weights.tolist(),
        "bias": float(bias)
    }
    
    # Create sv json
    sv_dict = {
        "seperable": separable,  # Note: keeping the misspelling as per the example
        "support_vectors": support_vectors if separable else []
    }
    
    # Generate output filenames
    base_name = filename.split('.')[0]
    label = base_name.split('_')[1]  # Assuming file name format is "train_<label>.csv"
    weight_file = f"weight_{label}.json"
    sv_file = f"sv_{label}.json"
    
    # Save json files
    with open(weight_file, 'w') as f:
        json.dump(weight_dict, f, indent=2)
    
    with open(sv_file, 'w') as f:
        json.dump(sv_dict, f, indent=2)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python svm.py <training_data.csv>")
        sys.exit(1)
        
    training_file = sys.argv[1]
    process_dataset(training_file)
