from keras.models import load_model
from keras.callbacks import EarlyStopping
from model import create_model
from data_preparation import create_sequences
from feature_extraction import extract_features
from data_collection import collect_data


def train_model(model, X, y, epochs=100, batch_size=32):
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    model.fit(X, y, epochs=epochs, batch_size=batch_size,
              validation_split=0.2, callbacks=[es])


if __name__ == "__main__":
    closing_prices = collect_data()
    fourier_features = extract_features(closing_prices)
    X, y = create_sequences(fourier_features, closing_prices, seq_length=60)
    model = create_model((X.shape[1], X.shape[2]))
    train_model(model, X, y)
    model.save('trained_model.h5')  # saving the trained model
