import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import pickle
import os

# Function to collect data from each csv file
def collect_data(filename):
    # Read the data file
    data = pd.read_csv(f'./data/raw_data/{filename}.csv')

    return data

# Function to process data
def process_data(data):
    # separate scaler for target column
    target_scaler = MinMaxScaler(feature_range=(-1, 1))

    # scale all columns
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_scaled = pd.DataFrame(
        scaler.fit_transform(data), columns=data.columns)

    # scale target column with separate scaler
    data_scaled[data_scaled.columns] = target_scaler.fit_transform(data)

    
    # save the target scaler
    dir_name = './data/processed_data'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(os.path.join(dir_name, 'target_scaler.pkl'), 'wb') as file:
        pickle.dump(target_scaler, file)

    return data_scaled, target_scaler

def split_data(data):
    train_size = int(len(data) * 0.8)
    train, test = data[0:train_size], data[train_size:len(data)]
    return train, test

def main():
    # List of all data files to collect
    for i in range(15):
        # Collect data
        filename = f'simulated{i+1}'
        data = collect_data(filename)

        # Process data
        data_scaled, target_scaler = process_data(data)

        # Save processed data
        data_scaled.to_csv(f'./data/processed_data/{filename}_processed.csv', index=False)

        # Split data
        train, test = split_data(data_scaled)
        train.to_csv(f'./data/processed_data/{filename}_train.csv', index=False)
        test.to_csv(f'./data/processed_data/{filename}_test.csv', index=False)

        # Save target scaler
        with open(f'./data/processed_data/target_scaler_{i+1}.pkl', 'wb') as file:
            pickle.dump(target_scaler, file)

if __name__ == "__main__":
    main()
