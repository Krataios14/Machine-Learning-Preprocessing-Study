import os
import pickle
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib

# Function to prepare data for LSTM input


def prepare_data(data, lookback):
    X = []
    for i in range(len(data)-lookback-1):
        t = data[i:(i+lookback), :]
        X.append(t)
    return np.array(X)

# Function to reverse the scaling for a list of values


def reverse_scale(predictions):
    # Load the target_scaler used during training
    with open('./data/processed_data/target_scaler.pkl', 'rb') as file:
        target_scaler = pickle.load(file)
    # Perform the inverse transformation
    return target_scaler.inverse_transform(predictions)


def main():
    # Define the lookback period
    lookback = 60

    # Load the trained model
    model = load_model('./models/saved_models.h5')

    # Load the data scaler
    scaler = joblib.load('./models/scaler.pkl')

    # Read the data
    data = pd.read_csv('./data/processed_data/yahoo_stock_prices_test.csv')

    # Scale the data
    data_scaled = scaler.transform(data)

    # Prepare the data for LSTM input
    X_test = prepare_data(data_scaled, lookback)

    # Generate predictions
    predictions = model.predict(X_test)

    # Reverse the scaling for the predictions
    predictions = reverse_scale(predictions)

    # Write the predictions to a CSV file
    pd.DataFrame(predictions).to_csv('./results/predictions.csv')


if __name__ == "__main__":
    main()
