import pandas as pd
import random


def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    sp500_df = tables[0]
    return sp500_df['Symbol'].tolist()


def select_random_tickers(tickers, count=50):
    return random.sample(tickers, count)


def calculate_expected_returns(price_data):
    return price_data.mean()
