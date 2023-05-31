import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import joblib
import pickle
import os

# Function to prepare data for LSTM input


def prepare_data(data, lookback):
    X, y = [], []
    for i in range(len(data)-lookback-1):
        t = data[i:(i+lookback), 0]
        X.append(t)
        y.append(data[i + lookback, 0])
    return np.array(X), np.array(y)


def create_and_train_model(X_train, y_train, lookback, epochs):
    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(lookback, 1)))
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

    return model


def main():
    # Define the lookback period and number of epochs
    lookback = 60
    epochs = 500

    # iterate through each training dataset
    for i in range(15):
        filename = f'simulated{i+1}'
        train_file = f'./data/processed_data/{filename}_train.csv'

        # check if the file exists
        if not os.path.exists(train_file):
            continue

        # Read the training data
        data = pd.read_csv(train_file)

        # Load the scaler
        with open(f'./data/processed_data/target_scaler_{i+1}.pkl', 'rb') as file:
            scaler = pickle.load(file)

        # Prepare the data for LSTM input
        data_scaled = scaler.transform(data)
        X_train, y_train = prepare_data(data_scaled, lookback)

        # Create and train the model for each dataset
        model = create_and_train_model(X_train, y_train, lookback, epochs)

        # Save the model for each dataset
        model.save(f'./models/saved_model_{i+1}.h5')
        print(f'model {i+1} saved')

    print('training completed')


if __name__ == "__main__":
    main()
