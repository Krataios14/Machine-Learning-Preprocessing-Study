import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the raw data
raw_data = pd.read_csv('./data/raw_data/yahoo_stock_prices.csv')
plot_file = './results/graph.png'

raw_dates = pd.to_datetime(raw_data['Date'])
raw_values = raw_data['Close']

plt.figure(figsize=(12, 6))
# Plot raw data
plt.plot(raw_dates, raw_values, label='Actual')

# Load and plot the predictions from the real dataset
# predictions_real = pd.read_csv('./results/predictions_real.csv', usecols=[1])
# prediction_start_date_real = raw_dates.iloc[-1] + pd.Timedelta(days=1)
# prediction_dates_real = pd.date_range(
#     start=prediction_start_date_real, periods=len(predictions_real))
# plt.plot(prediction_dates_real, predictions_real, label='Predicted Real')

# Iterate over each simulated prediction dataset
for i in range(1, 16):
    filename = f'predictions_simulated{i}.csv'
    prediction_file = f'./results/{filename}'

    # check if the file exists
    if not os.path.exists(prediction_file):
        continue

    # Load the predictions
    predictions = pd.read_csv(prediction_file, usecols=[1])

    # Set prediction_dates based on your prediction start date and the number of predictions
    prediction_start_date = prediction_dates_real.iloc[-1] + pd.Timedelta(days=1)
    prediction_dates = pd.date_range(
        start=prediction_start_date, periods=len(predictions))

    # Plot predictions
    plt.plot(prediction_dates, predictions, label=f'Predicted Simulated {i}')

plt.title('Historical and Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
plt.savefig(plot_file)
