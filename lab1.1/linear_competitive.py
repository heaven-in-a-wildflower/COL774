import numpy as np
import pandas as pd
import sys
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from category_encoders import MEstimateEncoder
from sklearn.linear_model import HuberRegressor
from sklearn.impute import SimpleImputer

pd.set_option('future.no_silent_downcasting', True)

train_file = sys.argv[1]
test_file = sys.argv[2]
output_file = sys.argv[3]

# Load the data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
y_test=pd.read_csv("test_pred.csv")

# Add a column of ones for the intercept term
train_data.insert(0, 'intercept', 1)
test_data.insert(0, 'intercept', 1)

# Define the target column and the list of categorical columns
target_column = 'Total Costs'  
categorical_columns = train_data.columns  # Replace with your categorical column names

# Initialize PolynomialFeatures for degrees 2 and 3
poly_3 = PolynomialFeatures(degree=3, include_bias=False)

# Apply target encoding, smoothing, and polynomial feature addition
for col in categorical_columns:
    if col == 'Total Costs':
        continue

    # Initialize MEstimateEncoder with the specified smoothing factor
    encoder = MEstimateEncoder(cols=[col], m=0.1)
    
    # Fit the encoder and transform training data
    train_data[col + '_encoded'] = encoder.fit_transform(train_data[[col]], train_data[target_column])
    
    # Apply the same encoding to the test data
    test_data[col + '_encoded'] = encoder.transform(test_data[[col]])
    
    # Optionally, fill NaN values in the test set if any category is missing in the training set
    test_data[col + '_encoded'].fillna(train_data[col + '_encoded'].mean())

    # Apply polynomial features of degree 3
    X_train_poly_3 = poly_3.fit_transform(train_data[[col + '_encoded']])
    X_test_poly_3 = poly_3.transform(test_data[[col + '_encoded']])
    
    poly_features_train_3 = pd.DataFrame(X_train_poly_3, columns=[f"{col}_encoded_poly3_{i}" for i in range(X_train_poly_3.shape[1])])
    poly_features_test_3 = pd.DataFrame(X_test_poly_3, columns=[f"{col}_encoded_poly3_{i}" for i in range(X_test_poly_3.shape[1])])
    
    # Concatenate polynomial features with the existing features
    train_data = pd.concat([train_data, poly_features_train_3], axis=1)
    test_data = pd.concat([test_data, poly_features_test_3], axis=1)

# Optional: Reset index if needed
train_data.reset_index(drop=True, inplace=True)
test_data.reset_index(drop=True, inplace=True)

# Ensure that the target column is not included in the features
X_train = train_data.drop(columns=[target_column])
y_train = train_data[target_column]
X_test = test_data

# Initialize an imputer with the mean strategy
imputer = SimpleImputer(strategy='mean')

# Fit the imputer on the training data
imputer.fit(X_train)

# Transform both the training and test data
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

huber = HuberRegressor(epsilon=1.35, alpha=0.1)

huber.fit(X_train, y_train)

y_pred=huber.predict(X_test)

y_pred = np.asarray(y_pred).ravel()
y_test = np.asarray(y_test).ravel()

# Compute the absolute errors
errors = np.abs(y_pred - y_test)

# Sort errors in ascending order
sorted_errors = np.sort(errors)

# Number of test samples
n = len(sorted_errors)

# Calculate the number of samples to include in the best 90%
n_90 = int(0.9 * n)

# Select the best 90% errors
top_90_errors = sorted_errors[:n_90]

# Compute the RMSE for these top 90% errors
rmse_top_90 = np.sqrt(np.mean(top_90_errors**2))
rmse=np.sqrt(np.mean(errors**2))

print(rmse_top_90)

np.savetxt(output_file, y_pred)