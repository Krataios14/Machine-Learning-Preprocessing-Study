import pandas as pd
import numpy as np
from scipy.fftpack import fft

def Fourier_transform(data, column_name):
    # Apply Fourier Transform on the specified column
    fourier_transform = fft(data[column_name].values)

    # Only keep the positive part of the spectrum
    half_spectrum = fourier_transform[:len(fourier_transform)//2]

    # Get the absolute values to have the magnitude of the frequencies
    magnitudes = np.abs(half_spectrum)

    return magnitudes


def main():
    # Read the processed stock price data
    data = pd.read_csv('./data/processed_data/yahoo_stock_prices_processed.csv')

    # Apply Fourier Transform on 'Close' column
    Fourier_features = Fourier_transform(data, 'Close')

    # Convert the features into a DataFrame for easier handling
    Fourier_df = pd.DataFrame(Fourier_features, columns=['Fourier_Features'])

    # Save Fourier features
    Fourier_df.to_csv('./data/features/Fourier_features.csv', index=False)


if __name__ == "__main__":
    main()
