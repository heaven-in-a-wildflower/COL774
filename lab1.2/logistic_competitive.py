import numpy as np
import sys
import pandas as pd
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

def calculate_quantitative_fraction_bins(df, category, bins, labels):
    df[f'{category} Binned'] = pd.cut(df[category], bins=bins, labels=labels, right=False)
    gender_distribution = df.groupby([f'{category} Binned', 'Gender'], observed=False).size().unstack().fillna(0)
    gender_fraction = gender_distribution.div(gender_distribution.sum(axis=1), axis=0)
    bins = {
        '0.0 to 0.05': [],
        '0.05 to 0.1': [],
        '0.1 to 0.15': [],
        '0.15 to 0.2': [],
        '0.2 to 0.25': [],
        '0.25 to 0.3': [],
        '0.3 to 0.35': [],
        '0.35 to 0.4': [],
        '0.4 to 0.45': [],
        '0.45 to 0.5': [],
        '0.5 to 0.55': [],
        '0.55 to 0.6': [],
        '0.6 to 0.65': [],
        '0.65 to 0.7': [],
        '0.7 to 0.75': [],
        '0.75 to 0.8': [],
        '0.8 to 0.85': [],
        '0.85 to 0.9': [],
        '0.9 to 0.95': [],
        '0.95 to 1.0': []
    }
    
    for value, row in gender_fraction.iterrows():
        female_fraction = row[-1]  
        if 0 <= female_fraction < 0.05:
            bins['0.0 to 0.05'].append(value)
        elif 0.05 <= female_fraction < 0.1:
            bins['0.05 to 0.1'].append(value)
        elif 0.1 <= female_fraction < 0.15:
            bins['0.1 to 0.15'].append(value)
        elif 0.15 <= female_fraction < 0.2:
            bins['0.15 to 0.2'].append(value)
        elif 0.2 <= female_fraction < 0.25:
            bins['0.2 to 0.25'].append(value)
        elif 0.25 <= female_fraction < 0.3:
            bins['0.25 to 0.3'].append(value)
        elif 0.3 <= female_fraction < 0.35:
            bins['0.3 to 0.35'].append(value)
        elif 0.35 <= female_fraction < 0.4:
            bins['0.35 to 0.4'].append(value)
        elif 0.4 <= female_fraction < 0.45:
            bins['0.4 to 0.45'].append(value)
        elif 0.45 <= female_fraction < 0.5:
            bins['0.45 to 0.5'].append(value)
        elif 0.5 <= female_fraction < 0.55:
            bins['0.5 to 0.55'].append(value)
        elif 0.55 <= female_fraction < 0.6:
            bins['0.55 to 0.6'].append(value)
        elif 0.6 <= female_fraction < 0.65:
            bins['0.6 to 0.65'].append(value)
        elif 0.65 <= female_fraction < 0.7:
            bins['0.65 to 0.7'].append(value)
        elif 0.7 <= female_fraction < 0.75:
            bins['0.7 to 0.75'].append(value)
        elif 0.75 <= female_fraction < 0.8:
            bins['0.75 to 0.8'].append(value)
        elif 0.8 <= female_fraction < 0.85:
            bins['0.8 to 0.85'].append(value)
        elif 0.85 <= female_fraction < 0.9:
            bins['0.85 to 0.9'].append(value)
        elif 0.9 <= female_fraction < 0.95:
            bins['0.9 to 0.95'].append(value)
        elif 0.95 <= female_fraction <= 1.0:
            bins['0.95 to 1.0'].append(value)
    
    return bins

def calculate_categorical_fraction_bins(df, category):
    gender_distribution = df.groupby([category, 'Gender']).size().unstack().fillna(0)
    gender_fraction = gender_distribution.div(gender_distribution.sum(axis=1), axis=0)
    gender_fraction_sorted = gender_fraction.sort_values(by=-1, ascending=False)
    bins = {
        '0.0 to 0.05': [],
        '0.05 to 0.1': [],
        '0.1 to 0.15': [],
        '0.15 to 0.2': [],
        '0.2 to 0.25': [],
        '0.25 to 0.3': [],
        '0.3 to 0.35': [],
        '0.35 to 0.4': [],
        '0.4 to 0.45': [],
        '0.45 to 0.5': [],
        '0.5 to 0.55': [],
        '0.55 to 0.6': [],
        '0.6 to 0.65': [],
        '0.65 to 0.7': [],
        '0.7 to 0.75': [],
        '0.75 to 0.8': [],
        '0.8 to 0.85': [],
        '0.85 to 0.9': [],
        '0.9 to 0.95': [],
        '0.95 to 1.0': []
    }
    for value, row in gender_fraction_sorted.iterrows():
        female_fraction = row[-1]  
        if 0 <= female_fraction < 0.05:
            bins['0.0 to 0.05'].append(value)
        elif 0.05 <= female_fraction < 0.1:
            bins['0.05 to 0.1'].append(value)
        elif 0.1 <= female_fraction < 0.15:
            bins['0.1 to 0.15'].append(value)
        elif 0.15 <= female_fraction < 0.2:
            bins['0.15 to 0.2'].append(value)
        elif 0.2 <= female_fraction < 0.25:
            bins['0.2 to 0.25'].append(value)
        elif 0.25 <= female_fraction < 0.3:
            bins['0.25 to 0.3'].append(value)
        elif 0.3 <= female_fraction < 0.35:
            bins['0.3 to 0.35'].append(value)
        elif 0.35 <= female_fraction < 0.4:
            bins['0.35 to 0.4'].append(value)
        elif 0.4 <= female_fraction < 0.45:
            bins['0.4 to 0.45'].append(value)
        elif 0.45 <= female_fraction < 0.5:
            bins['0.45 to 0.5'].append(value)
        elif 0.5 <= female_fraction < 0.55:
            bins['0.5 to 0.55'].append(value)
        elif 0.55 <= female_fraction < 0.6:
            bins['0.55 to 0.6'].append(value)
        elif 0.6 <= female_fraction < 0.65:
            bins['0.6 to 0.65'].append(value)
        elif 0.65 <= female_fraction < 0.7:
            bins['0.65 to 0.7'].append(value)
        elif 0.7 <= female_fraction < 0.75:
            bins['0.7 to 0.75'].append(value)
        elif 0.75 <= female_fraction < 0.8:
            bins['0.75 to 0.8'].append(value)
        elif 0.8 <= female_fraction < 0.85:
            bins['0.8 to 0.85'].append(value)
        elif 0.85 <= female_fraction < 0.9:
            bins['0.85 to 0.9'].append(value)
        elif 0.9 <= female_fraction < 0.95:
            bins['0.9 to 0.95'].append(value)
        elif 0.95 <= female_fraction <= 1.0:
            bins['0.95 to 1.0'].append(value)
    
    return bins

def apply_categorical_bins(df, category_bins):
    columns_to_remove = [
        'Operating Certificate Number', 'Permanent Facility Id', 
        'Total Costs', 'Length of Stay', 'Birth Weight'
    ]
    original_columns = set()
    for category, bins in category_bins.items():
        if category in df.columns:
            original_columns.add(category)
            for bin_range, values in bins.items():
                new_column_name = f"{category}_{bin_range.replace(' ', '')}"
                df[new_column_name] = df[category].apply(lambda x: 1 if x in values else 0)
    df.drop(columns=original_columns, inplace=True)
    df.drop(columns=[col for col in columns_to_remove if col in df.columns], inplace=True)
    if 'Gender' in df.columns:
        gender_col = df.pop('Gender')
        df['Gender'] = gender_col
    
    return df

def apply_quantitative_bins(df, category, fraction_bins):
    one_hot_df = pd.DataFrame()
    for bin_range, values in fraction_bins.items():
        new_column_name = f"{category}_{bin_range.replace(' ', '').replace('to', '_')}"
        one_hot_df[new_column_name] = df[f'{category} Binned'].apply(lambda x: 1 if x in values else 0)
    return one_hot_df

def load_data(file):
    data = pd.read_csv(file)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def predict(X, w):
    return sigmoid(np.dot(X, w))

def cauchy_loss(w, X, y_true, c):
    y_pred = predict(X, w)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    error = y_pred - y_true
    return np.mean(np.log(1 + (error/c)**2))

def cauchy_loss_gradient(w, X, y_true, c):
    n_samples = X.shape[0]
    y_pred = predict(X, w)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    error = y_pred - y_true
    gradient = (2 / n_samples) * np.dot(X.T, (error / (c**2 + error**2)) * y_pred * (1 - y_pred))
    return gradient

def ternary_search(X, y, w, g, eta_0, freq,c,max_iter=20):
    eta_l=0
    eta_h=eta_0 
    while cauchy_loss(w,X,y,c)>cauchy_loss(w-eta_h*g,X,y,c):
        eta_h*=2
    for _ in range(max_iter):
        eta_1=(2*eta_l+eta_h)/3
        eta_2=(eta_l+2*eta_h)/3

        if cauchy_loss(w-eta_1*g,X,y,c)>cauchy_loss(w-eta_2*g,X,y,c):
            eta_l=eta_1
        else:
            eta_h=eta_2
    eta=(eta_l+eta_h)/2
    return eta

def gradient_descent(X, y, w, freq, c, rate_params, length, epochs, strategy):
    eta_0 = rate_params[0]
    k = rate_params[1] if strategy == 2 else 0 
    n = X.shape[0]
    for epoch in range(epochs):
        for i in range(0, n, length):
            X_batch = X[i:i+length]
            y_batch = y[i:i+length]

            if strategy == 3:
                g = cauchy_loss_gradient(w,X_batch,y_batch,c)
                eta = ternary_search(X_batch, y_batch, w, g, eta_0, freq,c)
                rate = eta
            elif strategy == 2:
                rate = eta_0 / (1 + k * epoch)
            else:
                rate = eta_0

            batch_loss = cauchy_loss(w,X_batch,y_batch,c)
            log_message = f"Epoch Number: {epoch + 1}, Batch Number: {i // length + 1}, Batch Loss(before updating weights): {batch_loss:.6f}"
            if i // length + 1 == 1:
                print(log_message)

            if strategy == 3:
                w -= rate * g
    return w

def preprocess_dataset(train_file, test_file):
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    categories = [
        "APR MDC Code", "APR DRG Code", "Hospital Service Area", "Hospital County", "Facility Name", "Age Group",
        "Zip Code - 3 digits", "Race", "Ethnicity", "Type of Admission",
        "Patient Disposition", "CCSR Diagnosis Code", "CCSR Procedure Code",
        "APR Severity of Illness Description", "APR Risk of Mortality",
        "APR Medical Surgical Description", "Payment Typology 1",
        "Payment Typology 2", "Payment Typology 3", "Emergency Department Indicator"
    ]

    los_bins = [0, 5, 10, 15, 20, 25, 30, float('inf')]
    los_labels = [f'{los_bins[i]} to {los_bins[i+1]}' for i in range(len(los_bins) - 1)]

    los_fraction_bins = calculate_quantitative_fraction_bins(df_train, 'Length of Stay', los_bins, los_labels)

    df_train_one_hot = pd.concat([
        apply_quantitative_bins(df_train, 'Length of Stay', los_fraction_bins)
    ], axis=1)

    df_train = df_train.drop(columns=['Length of Stay','Length of Stay Binned'])
    df_train = pd.concat([df_train, df_train_one_hot], axis=1)

    df_test['Length of Stay Binned'] = pd.cut(df_test['Length of Stay'], bins=los_bins, labels=los_labels, right=False)

    df_test_one_hot = pd.concat([
        apply_quantitative_bins(df_test, 'Length of Stay', los_fraction_bins)
    ], axis=1)

    df_test = df_test.drop(columns=['Length of Stay','Length of Stay Binned'])
    df_test = pd.concat([df_test, df_test_one_hot], axis=1)

    category_bins = {}
    for category in categories:
        category_bins[category] = calculate_categorical_fraction_bins(df_train, category)

    df_train = apply_categorical_bins(df_train, category_bins)
    df_test = apply_categorical_bins(df_test, category_bins)

    return df_train, df_test

def main(train_file, test_file, output_file):
    df_train,df_test = preprocess_dataset(train_file,test_file)
    X_train = df_train.values[:, :-1]
    y_train = df_train.values[:, -1]
    y_train = (y_train + 1)/2
    X_test = df_test.values

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_train = np.insert(X_train, 0, np.ones(X_train.shape[0]), axis=1)
    X_test = scaler.transform(X_test)
    X_test = np.insert(X_test, 0, np.ones(X_test.shape[0]), axis=1)

    freq = {0: np.sum(y_train == 0), 1: np.sum(y_train == 1)}  
    w = np.zeros(X_train.shape[1])

    best_c = 1.2
    w = gradient_descent(X_train, y_train, w, freq, best_c, rate_params=[np.float64('1e-6')], length=X_train.shape[0], epochs=20, strategy=3)
    y_pred = predict(X_test, w)
    y_pred = 2 * y_pred - 1
    y_pred = np.where(y_pred > 0, 1, -1)
    y_pred = y_pred.astype(int)
    np.savetxt(output_file, y_pred, fmt='%.0f', delimiter="\n")

if __name__ == "__main__":
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    output_file = sys.argv[3]
    main(train_file, test_file, output_file)