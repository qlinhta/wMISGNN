import yfinance as yf
import pandas as pd
from tqdm import tqdm


def fetch_data(ticker, start_date, end_date):
    stock_data = yf.Ticker(ticker)
    return stock_data.history(period='1d', start=start_date, end=end_date)['Close']


def fetch_data_for_tickers(tickers, start_date, end_date):
    price_data = {}
    for ticker in tqdm(tickers, desc="Fetching Data"):
        price_data[ticker] = fetch_data(ticker, start_date, end_date)
    return pd.DataFrame(price_data)


def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    sp500_df = tables[0]
    return sp500_df['Symbol'].tolist()


if __name__ == "__main__":
    sp500_tickers = get_sp500_tickers()
    data = fetch_data_for_tickers(sp500_tickers, "2002-01-01", "2022-12-31")
    data.to_csv('sp500_data.csv', index=True)
