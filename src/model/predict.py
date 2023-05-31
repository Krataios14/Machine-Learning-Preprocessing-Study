import os
import pickle
import numpy as np
import pandas as pd
from keras.models import load_model

# Function to prepare data for LSTM input


def prepare_data(data, lookback):
    X = []
    for i in range(len(data)-lookback-1):
        t = data[i:(i+lookback), 0]
        X.append(t)
    return np.array(X)

# Function to reverse the scaling for a list of values


def reverse_scale(scaler, predictions):
    return scaler.inverse_transform(predictions)


def main():
    # Define the lookback period
    lookback = 60

    # Iterate over each test dataset
    for i in range(15):
        filename = f'simulated{i+1}'
        test_file = f'./data/processed_data/{filename}_test.csv'
        model_file = f'./models/saved_model_{i+1}.h5'

        # check if the file exists
        if not os.path.exists(test_file) or not os.path.exists(model_file):
            continue

        # Load the trained model
        model = load_model(model_file)

        # Read the test data
        data = pd.read_csv(test_file)

        # Load the scaler
        with open(f'./data/processed_data/target_scaler_{i+1}.pkl', 'rb') as file:
            scaler = pickle.load(file)

        # Prepare the data for LSTM input
        data_scaled = scaler.transform(data)
        X_test = prepare_data(data_scaled, lookback)

        # Generate predictions
        predictions = model.predict(X_test)

        # Reverse the scaling for the predictions
        predictions = reverse_scale(scaler, predictions)

        # Write the predictions to a CSV file
        pd.DataFrame(predictions).to_csv(f'./results/predictions_{i+1}.csv')


if __name__ == "__main__":
    main()
