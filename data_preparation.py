import numpy as np

def create_sequences(fourier_features, closing_prices, seq_length):
    X, y = [], []

    for i in range(seq_length, len(fourier_features)):
        X.append(fourier_features[i-seq_length:i])
        y.append(closing_prices[i])

    return np.array(X), np.array(y)

def split_data(X, y, test_size=0.2):
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    # example usage:
    X, y = create_sequences(np.random.rand(100, 2), np.random.rand(100), seq_length=10)
    X_train, y_train, X_test, y_test = split_data(X, y, test_size=0.2)
    print(X_train, y_train, X_test, y_test)
