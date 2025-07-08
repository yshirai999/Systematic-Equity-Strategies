import os
import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
from datetime import datetime
from scipy.io import savemat
from dotenv import load_dotenv

# Load .env file to access FRED API key
load_dotenv()
FRED_API_KEY = os.getenv('FRED_API_KEY')

def download_macro_data(start='2010-01-01', end=None):
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')

    # 1. Download Yahoo Finance data
    tickers = {
        'VIX': '^VIX',
        'Oil': 'CL=F',
        'SP500': '^GSPC'
    }

    df_yahoo = {}
    for name, ticker in tickers.items():
        data = yf.download(ticker, start=start, end=end)['Adj Close']
        df_yahoo[name] = data.rename(name)

    # 2. Download Treasury rates from FRED
    fred_symbols = {
        '10Y': 'DGS10',
        '2Y': 'DGS2',
        # Optional: 'SKEW': 'SKEW'
    }

    df_fred = {}
    for name, symbol in fred_symbols.items():
        df = web.DataReader(symbol, 'fred', start, end, api_key=FRED_API_KEY)
        df_fred[name] = df.rename(columns={symbol: name})

    # 3. Merge all dataframes
    df_all = pd.concat([*df_yahoo.values(), *df_fred.values()], axis=1)
    df_all = df_all.dropna()

    # 4. Save as CSV
    csv_path = os.path.join(os.path.dirname(__file__), 'macro_factors.csv')
    df_all.to_csv(csv_path)

    # 5. Save as .mat file
    mat_path = os.path.join(os.path.dirname(__file__), 'macro_factors.mat')
    mat_dict = {col: df_all[col].values for col in df_all.columns}
    mat_dict['dates'] = df_all.index.strftime('%Y-%m-%d').to_list()
    savemat(mat_path, mat_dict)

    print("Data saved to:")
    print(f"- {csv_path}")
    print(f"- {mat_path}")

if __name__ == '__main__':
    download_macro_data()
