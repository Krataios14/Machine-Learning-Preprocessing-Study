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
    # Remove date column if it exists
    if 'Date' in data.columns:
        data = data.drop(['Date'], axis=1)

    # separate scaler for target column
    target_scaler = MinMaxScaler(feature_range=(-1, 1))

    # scale all columns
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_scaled = pd.DataFrame(
        scaler.fit_transform(data), columns=data.columns)

    # scale target column with separate scaler
    data_scaled['Close'] = target_scaler.fit_transform(data[['Close']])
    dir_name = '.\\data\\processed_data\\'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    # save the target scaler
    with open(os.path.join(dir_name, 'target_scaler.pkl'), 'wb') as file:
        pickle.dump(target_scaler, file)


    return data_scaled



def main():
    # List of all data files to collect
    data_files = ['yahoo_stock_prices', ]


# 'technical_indicators', 'fundamental_data',
# 'market_sentiment', 'economic_indicators', 'commodity_prices'
    for filename in data_files:
        # Collect data
        data = collect_data(filename)

        # Process data
        data_scaled = process_data(data)

        # Save processed data
        data_scaled.to_csv(
            f'./data/processed_data/{filename}_processed.csv', index=False)


    def split_data(data):
        train_size = int(len(data) * 0.8)
        train, test = data[0:train_size], data[train_size:len(data)]
        return train, test


    train, test = split_data(data_scaled)
    train.to_csv(f'./data/processed_data/{filename}_train.csv', index=False)
    test.to_csv(f'./data/processed_data/{filename}_test.csv', index=False)


if __name__ == "__main__":
    main()
