import numpy as np
from keras.models import load_model
from data_preparation import create_sequences
from feature_extraction import extract_features
from data_collection import collect_data


def make_prediction(model, X):
    prediction = model.predict(X)
    return prediction


if __name__ == "__main__":
    closing_prices = collect_data()
    fourier_features = extract_features(closing_prices)
    X, _ = create_sequences(fourier_features, closing_prices, seq_length=60)
    model = load_model('trained_model.h5')  # load the trained model
    prediction = make_prediction(model, X)
    print(prediction)
