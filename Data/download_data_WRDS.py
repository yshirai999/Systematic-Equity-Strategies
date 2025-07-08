import wrds
import pandas as pd
from scipy.io import savemat
import os

def download_from_wrds(start_date='2010-01-01', save_path='data'):
    # 1. Connect to WRDS (you'll be prompted to enter credentials once)
    print("Connecting to WRDS...")
    db = wrds.Connection()

    print("Downloading VIX from FRED...")
    vix = db.raw_sql(f"""
        SELECT date, vix_close AS vix
        FROM fred.vix
        WHERE date >= '{start_date}'
    """)

    print("Downloading WTI Oil from FRED...")
    oil = db.raw_sql(f"""
        SELECT date, oil_price
        FROM fred.oil_price
        WHERE date >= '{start_date}'
    """)

    print("Downloading 10-Year Treasury Yield from FRED...")
    treasury = db.raw_sql(f"""
        SELECT date, dgs10 AS treasury_10y
        FROM fred.dgs10
        WHERE date >= '{start_date}'
    """)

    print("Merging and cleaning data...")
    df = vix.merge(oil, on='date', how='inner').merge(treasury, on='date', how='inner')
    df = df.dropna()
    df = df.sort_values('date')

    # Save as CSV
    csv_path = os.path.join(save_path, 'macro_factors_wrds.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to: {csv_path}")

    # Save as MAT
    mat_path = os.path.join(save_path, 'macro_factors_wrds.mat')
    mat_dict = {col: df[col].values for col in df.columns if col != 'date'}
    mat_dict['dates'] = df['date'].astype(str).to_list()
    savemat(mat_path, mat_dict)
    print(f"Saved MAT file to: {mat_path}")

    print("Done!")

if __name__ == '__main__':
    download_from_wrds()
