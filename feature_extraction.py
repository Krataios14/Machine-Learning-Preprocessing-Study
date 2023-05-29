import numpy as np
from scipy.signal import get_window
from data_collection import collect_data


def extract_features(closing_prices):
    window = get_window(('kaiser', 14), len(closing_prices))
    windowed_signal = closing_prices * window
    coefficients = np.fft.rfft(windowed_signal)
    max_terms = 10000000
    coefficients = coefficients[:max_terms]
    real_parts = coefficients.real
    imag_parts = coefficients.imag
    fourier_features = np.column_stack((real_parts, imag_parts))
    return fourier_features


if __name__ == "__main__":
    closing_prices = collect_data()
    fourier_features = extract_features(closing_prices)
    print(fourier_features)
