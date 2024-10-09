import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# Functions from your provided script

def calculate_quantitative_fraction_bins(df, category, bins, labels):
    df[f'{category} Binned'] = pd.cut(df[category], bins=bins, labels=labels, right=False)
    gender_distribution = df.groupby([f'{category} Binned', 'Gender'], observed=False).size().unstack().fillna(0)
    gender_fraction = gender_distribution.div(gender_distribution.sum(axis=1), axis=0)
    bins_dict = {
        '0.0 to 0.05': [], '0.05 to 0.1': [], '0.1 to 0.15': [], '0.15 to 0.2': [],
        '0.2 to 0.25': [], '0.25 to 0.3': [], '0.3 to 0.35': [], '0.35 to 0.4': [],
        '0.4 to 0.45': [], '0.45 to 0.5': [], '0.5 to 0.55': [], '0.55 to 0.6': [],
        '0.6 to 0.65': [], '0.65 to 0.7': [], '0.7 to 0.75': [], '0.75 to 0.8': [],
        '0.8 to 0.85': [], '0.85 to 0.9': [], '0.9 to 0.95': [], '0.95 to 1.0': []
    }
    
    for value, row in gender_fraction.iterrows():
        female_fraction = row[-1]
        for bin_range in bins_dict:
            lower, upper = map(float, bin_range.split(' to '))
            if lower <= female_fraction < upper:
                bins_dict[bin_range].append(value)
                break
    
    return bins_dict

def calculate_categorical_fraction_bins(df, category):
    gender_distribution = df.groupby([category, 'Gender']).size().unstack().fillna(0)
    gender_fraction = gender_distribution.div(gender_distribution.sum(axis=1), axis=0)
    gender_fraction_sorted = gender_fraction.sort_values(by=-1, ascending=False)
    bins_dict = {
        '0.0 to 0.05': [], '0.05 to 0.1': [], '0.1 to 0.15': [], '0.15 to 0.2': [],
        '0.2 to 0.25': [], '0.25 to 0.3': [], '0.3 to 0.35': [], '0.35 to 0.4': [],
        '0.4 to 0.45': [], '0.45 to 0.5': [], '0.5 to 0.55': [], '0.55 to 0.6': [],
        '0.6 to 0.65': [], '0.65 to 0.7': [], '0.7 to 0.75': [], '0.75 to 0.8': [],
        '0.8 to 0.85': [], '0.85 to 0.9': [], '0.9 to 0.95': [], '0.95 to 1.0': []
    }
    
    for value, row in gender_fraction_sorted.iterrows():
        female_fraction = row[-1]
        for bin_range in bins_dict:
            lower, upper = map(float, bin_range.split(' to '))
            if lower <= female_fraction < upper:
                bins_dict[bin_range].append(value)
                break
    
    return bins_dict

def apply_categorical_bins(df, category_bins):
    columns_to_remove = [
        'Operating Certificate Number', 'Permanent Facility Id', 
        'Total Costs', 'Length of Stay', 'Birth Weight'
    ]
    original_columns = set()
    created_features = []
    
    for category, bins in category_bins.items():
        if category in df.columns:
            original_columns.add(category)
            for bin_range, values in bins.items():
                new_column_name = f"{category}_{bin_range.replace(' ', '')}"
                df[new_column_name] = df[category].apply(lambda x: 1 if x in values else 0)
                created_features.append(new_column_name)
    
    df.drop(columns=original_columns, inplace=True)
    df.drop(columns=[col for col in columns_to_remove if col in df.columns], inplace=True)
    
    if 'Gender' in df.columns:
        gender_col = df.pop('Gender')
        df['Gender'] = gender_col
    
    return df, created_features

def apply_quantitative_bins(df, category, fraction_bins):
    one_hot_df = pd.DataFrame()
    created_features = []
    
    for bin_range, values in fraction_bins.items():
        new_column_name = f"{category}_{bin_range.replace(' ', '').replace('to', '_')}"
        one_hot_df[new_column_name] = df[f'{category} Binned'].apply(lambda x: 1 if x in values else 0)
        created_features.append(new_column_name)
    
    return one_hot_df, created_features

def preprocess_dataset(train_file):
    df_train = pd.read_csv(train_file)

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

    df_train_one_hot, los_created_features = apply_quantitative_bins(df_train, 'Length of Stay', los_fraction_bins)
    df_train = df_train.drop(columns=['Length of Stay','Length of Stay Binned'])
    df_train = pd.concat([df_train, df_train_one_hot], axis=1)

    category_bins = {}
    created_features = []
    for category in categories:
        category_bins[category] = calculate_categorical_fraction_bins(df_train, category)
    
    df_train, cat_created_features = apply_categorical_bins(df_train, category_bins)
    
    created_features.extend(los_created_features)
    created_features.extend(cat_created_features)
    
    return df_train, created_features

def main(train_file):
    df_train, created_features = preprocess_dataset(train_file)
    
    # Saving the created features to created.txt
    with open('created.txt', 'w') as f:
        for feature in created_features:
            f.write(f"{feature}\n")
    
    # For selected.txt, you could write a strategy for selecting features here.
    # For simplicity, we'll assume that all created features are selected.
    with open('selected.txt', 'w') as f:
        for feature in created_features:
            f.write("1\n")  # 1 indicates the feature is selected

if __name__ == "__main__":
    train_file = 'train2.csv'
    main(train_file)
