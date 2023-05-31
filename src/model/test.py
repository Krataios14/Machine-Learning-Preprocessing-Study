import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import joblib
import pickle
import os

# Function to prepare data for LSTM input


def prepare_data(data, lookback):
    X, y = [], []
    for i in range(len(data)-lookback-1):
        t = data[i:(i+lookback), :]
        X.append(t)
        y.append(data[i + lookback, 0])
    return np.array(X), np.array(y)

# Function to reverse the scaling for a list of values


def reverse_scale(predictions):
    # Load the target_scaler used during training
    with open('./data/processed_data/target_scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    # Perform the inverse transformation
    return scaler.inverse_transform(predictions)





def main():
    # Define the lookback period
    lookback = 60

    # Load the trained model
    model = load_model('./models/saved_models.h5')

    # Read the test data
    data = pd.read_csv('./data/processed_data/yahoo_stock_prices_test.csv')

    # Load the scaler used during training
    scaler = joblib.load('./models/scaler.pkl')

    # Scale the data
    data_scaled = scaler.transform(data)

    # Prepare the data for LSTM input
    X_test, y_test = prepare_data(data_scaled, lookback)

    # Make predictions
    predictions = model.predict(X_test)

    # Reshape the prediction to 2D for reverse scaling
    predictions = np.reshape(predictions, (predictions.size, 1))

    # Reverse the scaling of the prediction and y_test
    predictions = reverse_scale(predictions)
    y_test = reverse_scale(y_test.reshape(-1, 1))


    # Compute evaluation metrics
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    # Save evaluation metrics
    pd.DataFrame([mse, mae], index=['MSE', 'MAE'], columns=[
                 'Value']).to_csv('./results/evaluation_metrics.csv')


if __name__ == "__main__":
    main()
