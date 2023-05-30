import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Function to calculate GBM parameters
def calculate_gbm_params(data):
    log_returns = np.log(1 + data['Close'].pct_change())
    u = log_returns.mean() # Mean of the logarithmic return
    var = log_returns.var() # Variance of the logarithmic return
    drift = u - (0.5 * var) # Drift
    stdev = log_returns.std() # Volatility
    return drift, stdev

# Function to simulate GBM
def simulate_gbm(start_price, drift, stdev, days, iterations):
    daily_returns = np.exp(drift + stdev * np.random.normal(0, 1, (days, iterations)))
    price_list = np.zeros_like(daily_returns)
    price_list[0] = start_price
    for t in range(1, days):
        price_list[t] = price_list[t - 1] * daily_returns[t]
    return price_list

# File paths
predictions_file = './results/predictions.csv'
historical_data_file = './data/raw_data/yahoo_stock_prices.csv'
simulation_file = './results/simulation.csv'
plot_file = './results/brownian_simulation.png'

# Check if files exist
if not os.path.exists(predictions_file):
    print(f"Error: {predictions_file} does not exist!")
    sys.exit(1)
if not os.path.exists(historical_data_file):
    print(f"Error: {historical_data_file} does not exist!")
    sys.exit(1)

# Load data
predictions = pd.read_csv(predictions_file)
historical_data = pd.read_csv(historical_data_file)

# Validate data
if 'Close' not in historical_data.columns:
    print("Error: 'Close' column not found in historical data!")
    sys.exit(1)

# Calculate GBM parameters
drift, stdev = calculate_gbm_params(historical_data)

# Number of paths for simulation
iterations = 100

# Simulate GBM
simulation = simulate_gbm(historical_data['Close'].iloc[-1], drift, stdev, len(predictions), iterations)

# Save the simulation to a CSV
np.savetxt(simulation_file, simulation, delimiter=",")

# Plot the first 10 paths
plt.figure(figsize=(10,6))
plt.plot(simulation[:, :10])
plt.title('Geometric Brownian Motion Simulation')
plt.xlabel('Time')
plt.ylabel('Price')
plt.savefig(plot_file)
