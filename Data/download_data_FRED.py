import os
import pandas_datareader.data as web
import pandas as pd
from datetime import datetime
from scipy.io import savemat
from dotenv import load_dotenv

def download_fred_data(start='2007-01-01', save_path='data'):
    # Load API key
    load_dotenv()
    fred_key = os.getenv("FRED_API_KEY")

    if fred_key is None:
        raise ValueError("FRED_API_KEY not found. Please set it in your .env file.")

    # Series to download
    series = {
        'vix': 'VIXCLS',
        'oil': 'DCOILWTICO',
        'rate_10y': 'DGS10',
        'rate_2y': 'DGS2'
    }

    end = datetime.today().strftime('%Y-%m-%d')

    df_all = pd.DataFrame()

    for name, code in series.items():
        print(f"Downloading {name} ({code})...")
        df = web.DataReader(code, 'fred', start, end, api_key=fred_key)
        df.columns = [name]
        df_all = pd.concat([df_all, df], axis=1)

    df_all = df_all.dropna().sort_index()

    # Save as CSV
    #csv_path = os.path.join(save_path, 'macro_factors_fred.csv')
    #df_all.to_csv(csv_path)
    #print(f"CSV saved to: {csv_path}")

    # Save as .mat
    mat_path = os.path.join(save_path, 'macro_factors_fred.mat')
    mat_dict = {col: df_all[col].values for col in df_all.columns}
    mat_dict['dates'] = df_all.index.strftime('%Y-%m-%d').to_list()
    savemat(mat_path, mat_dict)
    print(f"MAT file saved to: {mat_path}")

if __name__ == '__main__':
    download_fred_data()
