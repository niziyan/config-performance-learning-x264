import os
import json
import argparse
import numpy as np
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import data_loader

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-10))) * 100

def train_and_evaluate(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    return mae, rmse, mape, y_pred

def run_experiment(dataset_dir, csv_file, num_runs=30):
    print(f"Running linear regression on {dataset_dir}/{csv_file} for {num_runs} times...")
    
    all_metrics = {
        'MAE': [],
        'RMSE': [],
        'MAPE': []
    }
    
    for i in range(num_runs):
        X_train, X_test, y_train, y_test = data_loader.load_dataset(
            dataset_dir, csv_file, random_state=i
        )
        
        mae, rmse, mape, y_pred = train_and_evaluate(X_train, X_test, y_train, y_test)
        
        all_metrics['MAE'].append(mae)
        all_metrics['RMSE'].append(rmse)
        all_metrics['MAPE'].append(mape)
        
        if i == 0:
            first_predictions = y_pred.tolist()
    
    mae_mean = np.mean(all_metrics['MAE'])
    mae_std = np.std(all_metrics['MAE'])
    rmse_mean = np.mean(all_metrics['RMSE'])
    rmse_std = np.std(all_metrics['RMSE'])
    mape_mean = np.mean(all_metrics['MAPE'])
    mape_std = np.std(all_metrics['MAPE'])
    
    mae_formatted = f"{mae_mean:.2f}±{mae_std:.2f}"
    rmse_formatted = f"{rmse_mean:.2f}±{rmse_std:.2f}"
    mape_formatted = f"{mape_mean:.2f}±{mape_std:.2f}"
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    date_dir = datetime.datetime.now().strftime("%Y%m%d")
    log_dir = os.path.join("log", date_dir, "linear-regression")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"lr_{dataset_dir}_{csv_file}_{timestamp}.txt")
    
    with open(log_file, 'w') as f:
        f.write(f"dataset: {csv_file}\n")
        f.write(f"runs: {num_runs}\n")
        f.write(f"MAE: {mae_formatted}\n")
        f.write(f"RMSE: {rmse_formatted}\n")
        f.write(f"MAPE: {mape_formatted}\n")
    
    print(f"Results saved to {log_file}")
    print(f"MAE: {mae_formatted}")
    print(f"RMSE: {rmse_formatted}")
    print(f"MAPE: {mape_formatted}")

def main():
    parser = argparse.ArgumentParser(description='Train a linear regression model on a dataset.')
    parser.add_argument('--dataset', type=str, default='x264',
                        help='Dataset directory (default: x264)')
    parser.add_argument('--csv', type=str, default=None,
                        help='CSV file name (if None, all CSV files will be processed)')
    parser.add_argument('--runs', type=int, default=30,
                        help='Number of runs (default: 30)')
    
    args = parser.parse_args()
    
    datasets = data_loader.list_datasets()
    
    if args.dataset not in datasets:
        print(f"Dataset {args.dataset} not found. Available datasets: {list(datasets.keys())}")
        return
    
    if args.csv is None:
        for csv_file in datasets[args.dataset]:
            run_experiment(args.dataset, csv_file, args.runs)
    else:
        if args.csv in datasets[args.dataset]:
            run_experiment(args.dataset, args.csv, args.runs)
        else:
            print(f"CSV file {args.csv} not found in dataset {args.dataset}.")
            print(f"Available CSV files: {datasets[args.dataset]}")

if __name__ == "__main__":
    main()