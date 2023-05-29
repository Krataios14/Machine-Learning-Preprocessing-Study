import pandas as pd


def collect_data(file_name):
    data = pd.read_csv(file_name)
    closing_prices = data['Close']
    return closing_prices


if __name__ == "__main__":
    closing_prices = collect_data('TSLA.csv')
    print(closing_prices)
