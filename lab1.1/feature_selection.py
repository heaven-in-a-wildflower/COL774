import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import PolynomialFeatures
from category_encoders import MEstimateEncoder

pd.set_option('future.no_silent_downcasting', True)

train_file = sys.argv[1]
created_file = sys.argv[2]
selected_file = sys.argv[3]

# Load the training data
train_data = pd.read_csv(train_file)

# Define the target column and categorical columns
target_column = 'Total Costs'
categorical_columns = train_data.columns

# Initialize PolynomialFeatures for degree 3
poly_3 = PolynomialFeatures(degree=3, include_bias=False)
created_features = []

# Feature creation: Target encoding, Polynomial Features
for col in categorical_columns:
    if col == target_column:
        continue

    # Initialize MEstimateEncoder with the specified smoothing factor
    encoder = MEstimateEncoder(cols=[col], m=0.1)
    
    # Fit the encoder and transform training data
    train_data[col + '_encoded'] = encoder.fit_transform(train_data[[col]], train_data[target_column])
    created_features.append(f"{col}_encoded")
    
    # Apply polynomial features of degree 3
    X_train_poly_3 = poly_3.fit_transform(train_data[[col + '_encoded']])
    poly_features_3 = [f"{col}_encoded_poly3_{i}" for i in range(X_train_poly_3.shape[1])]
    created_features.extend(poly_features_3)

# Write created.txt
with open(created_file, 'w') as created:
    for feature in created_features:
        created.write(f"{feature}\n")

# Write selected.txt (all features are selected in this case)
with open(selected_file, 'w') as selected:
    for _ in created_features:
        selected.write("1\n")
