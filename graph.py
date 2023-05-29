import pandas as pd
import matplotlib.pyplot as plt

# Load the raw data
raw_data = pd.read_csv('./data/raw_data/yahoo_stock_prices.csv')

# Assume the 'Date' and 'Close' are the columns in your raw data
raw_dates = pd.to_datetime(raw_data['Date'])
raw_values = raw_data['Close']

# Load the predictions
predictions = pd.read_csv('./results/predictions.csv', usecols=[1])

# Set prediction_dates based on your prediction start date and the number of predictions

prediction_start_date = raw_dates.iloc[-1] + pd.Timedelta(days=1)
prediction_dates = pd.date_range(
    start=prediction_start_date, periods=len(predictions))

# Plot raw data
plt.figure(figsize=(12, 6))
plt.plot(raw_dates, raw_values, label='Actual')

# Plot predictions
plt.plot(prediction_dates, predictions, label='Predicted')

plt.title('Historical and Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
