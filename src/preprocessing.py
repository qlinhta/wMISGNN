import pandas as pd
import logging


def calculate_returns(price_data):
    return price_data.pct_change().dropna()


def calculate_correlations(returns_data):
    return returns_data.corr()


if __name__ == "__main__":
    df = pd.read_csv('../data/sp500_data.csv')
    df = df.dropna(axis=1)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    returns = calculate_returns(df)
    returns.to_csv('../data/sp500_returns.csv', index=True)
    correlations = calculate_correlations(returns)
    correlations.to_csv('../data/sp500_correlations.csv', index=True)
    # price data is in df to csv
    df.to_csv('../data/sp500_price.csv', index=True)
    logging.info("Done")
