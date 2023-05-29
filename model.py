from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping


def create_model(input_shape):
    model = Sequential()

    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))  # Add dropout with a rate of 0.2
    model.add(BatchNormalization())  # Add Batch normalization layer

    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))  # Add dropout with a rate of 0.2
    model.add(BatchNormalization())  # Add Batch normalization layer

    model.add(Dense(64, activation='relu'))  # Add additional Dense layer
    model.add(Dropout(0.2))  # Add dropout with a rate of 0.2
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


def train_model(model, train_x, train_y, epochs=10, batch_size=32):
    early_stop = EarlyStopping(
        monitor='val_loss', patience=2, verbose=1)  # Set up early stopping
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size,
              callbacks=[early_stop], validation_split=0.2)
