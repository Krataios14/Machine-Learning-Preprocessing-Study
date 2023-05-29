from data_collection import collect_data
from feature_extraction import extract_features
from data_preparation import create_sequences, split_data
from model import create_model
from train import train_model
from predict import make_prediction
from keras.models import load_model
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Collect the data
    closing_prices = collect_data('TSLA.csv')
    print("Data collection successful!")

    # Extract the Fourier features
    fourier_features = extract_features(closing_prices)
    print("Feature extraction successful!")

    # Prepare the data for LSTM and split into training and test sets
    seq_length = 60
    X, y = create_sequences(fourier_features, closing_prices, seq_length)
    X_train, y_train, X_test, y_test = split_data(X, y, test_size=0.2)
    print("Data preparation successful!")

    # Create and train the model
    model = create_model((X_train.shape[1], X_train.shape[2]))
    train_model(model, X_train, y_train)
    model.save('trained_model.h5')  # Save the trained model
    print("Model training and saving successful!")

    # Make predictions
    model = load_model('trained_model.h5')  # Load the trained model
    prediction = make_prediction(model, X_test)

    # Plot actual vs prediction
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, color='blue', label='Actual')
    plt.plot(prediction, color='red', label='Prediction')
    plt.title('Actual vs Predicted')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    print("Prediction making and plotting successful!")
