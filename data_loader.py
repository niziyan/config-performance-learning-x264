
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def list_datasets():
    datasets = {}
    for dataset_dir in os.listdir('datasets'):
        if os.path.isdir(os.path.join('datasets', dataset_dir)):
            datasets[dataset_dir] = []
            for csv_file in os.listdir(os.path.join('datasets', dataset_dir)):
                if csv_file.endswith('.csv'):
                    datasets[dataset_dir].append(csv_file)
    return datasets

def load_dataset(dataset_dir, csv_file, test_size=0.2, random_state=42):
    file_path = os.path.join('datasets', dataset_dir, csv_file)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    try:
        float(df.iloc[0, 0])
    except ValueError:
        df = pd.read_csv(file_path, skiprows=1, header=None)
    
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    datasets = list_datasets()
    print("Available datasets:")
    for dataset_dir, csv_files in datasets.items():
        print(f"{dataset_dir}:")
        for csv_file in csv_files:
            print(f"  - {csv_file}")