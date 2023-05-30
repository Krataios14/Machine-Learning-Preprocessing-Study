import pandas as pd

# Function to calculate Moving Average


def calculate_moving_average(data, window_size, column_name):
    # Calculate Moving Average using pandas 'rolling' function
    moving_average = data[column_name].rolling(window=window_size).mean()

    return moving_average


def main():
    # Define window size for Moving Average
    window_size_percentage = 0.05

    # Read the processed stock price data
    data = pd.read_csv('./data/processed_data/yahoo_stock_prices_processed.csv')
    window_size= int(len(data)*windwo_size_percentage)
    if window_size<1:
        window_size=1
    # Calculate Moving Average
    moving_average = calculate_moving_average(data, window_size, 'Close')
    moving_average.to_cvs('./data/features/Moving_Averages.cvs', header=['Moving_Average'])
    
    # Convert the features into a DataFrame for easier handling
    #     moving_average_df = pd.DataFrame(
    #         moving_average, columns=['Moving_Average'])
    # Save Moving Average features
    #     moving_average_df.to_csv('./data/features/Moving_Averages.csv', index=False)
    

if __name__ == "__main__":
    main()
