import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import joblib

# Function to prepare data for LSTM input


def prepare_data(data, lookback):
    X, y = [], []
    for i in range(len(data)-lookback-1):
        t = data[i:(i+lookback), :]
        X.append(t)
        y.append(data[i + lookback, 0])
    return np.array(X), np.array(y)


def main():
    # Define the lookback period and number of epochs
    lookback = 60
    epochs = 500

    # Read the training data
    data = pd.read_csv('./data/processed_data/yahoo_stock_prices_train.csv')

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    # Save the scaler
    joblib.dump(scaler, './models/scaler.pkl')

    # Prepare the data for LSTM input
    X_train, y_train = prepare_data(data_scaled, lookback)


    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True,
            input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

# Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

# Define the early stopping criteria
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

# Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=32,
          validation_split=0.2, callbacks=[es])

# Save the model
    model.save('./models/saved_models.h5')


if __name__ == "__main__":
    main()
