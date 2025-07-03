import pandas as pd
import requests
from io import StringIO
import os
from settings import config

DATA_DIR = config("DATA_DIR")
START_DATE = config("START_DATE")
END_DATE = config("END_DATE")

BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

FACTOR_FILES = {
    'EP_exDiv': 'Portfolios_Formed_on_E-P_Wout_Div.csv',
    'CFP_exDiv': 'Portfolios_Formed_on_CF-P_Wout_Div.csv',
    'DP_exDiv': 'Portfolios_Formed_on_D-P_Wout_Div.csv',
    'Net_Share_Issues': 'Portfolios_Formed_on_NI.csv',
    'Accruals': 'Portfolios_Formed_on_AC.csv',
    'Market_Beta': 'Portfolios_Formed_on_BETA.csv',
    'Variance': 'Portfolios_Formed_on_VAR.csv',
    'Residual_Variance': 'Portfolios_Formed_on_RESVAR.csv',
    'ST_Reversal': 'F-F_ST_Reversal_Factor.csv',
    'LT_Reversal': 'F-F_LT_Reversal_Factor.csv'
}

def auto_read_first_table_from_string(data_str):
    # Read the CSV data from a string, automatically identifying first data table
    lines = data_str.splitlines()

    # Identify start of first table (header line starts with a comma)
    data_start = None
    for idx, line in enumerate(lines):
        if line.startswith(',') or (',' in line and line.strip().split(',')[0] == ''):
            data_start = idx
            break

    if data_start is None:
        raise ValueError("Cannot find the start of data table")

    # Identify the end of the first table (empty line)
    data_end = None
    for idx in range(data_start + 1, len(lines)):
        if lines[idx].strip() == '':
            data_end = idx
            break

    if data_end is None:
        data_end = len(lines)

    # Extract the lines corresponding to the first table
    data_lines = lines[data_start:data_end]

    # Join the extracted lines and read into DataFrame
    first_table_str = '\n'.join(data_lines)
    df = pd.read_csv(StringIO(first_table_str), index_col=0)

    # Clean data
    df = df.dropna(how='all')
    df.index = pd.to_datetime(df.index, format='%Y%m', errors='coerce')
    df = df.dropna(axis=0, how='all')

    # Convert percentages to decimals
    df = df.apply(pd.to_numeric, errors='coerce') / 100
    
    df.index.name = 'date'
    
    return df

def download_and_save_all_factors(save_folder=DATA_DIR):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for factor, file_name in FACTOR_FILES.items():
        print(f"Downloading and processing {factor}...")
        try:
            # Download file content
            url = BASE_URL + file_name
            res = requests.get(url)
            res.raise_for_status()

            # Process downloaded content directly
            df = auto_read_first_table_from_string(res.text)
            
            # Save processed DataFrame to CSV
            save_path = os.path.join(save_folder, file_name)
            df.to_csv(save_path)
            print(f"Saved {factor} to {save_path}")

        except Exception as e:
            print(f"Failed to process {factor}: {e}")

if __name__ == "__main__":
    # download_and_save_all_factors()
    print(pd.read_csv(DATA_DIR/"F-F_LT_Reversal_Factor.csv"))
    print(pd.read_csv(DATA_DIR/"F-F_ST_Reversal_Factor.csv"))
    print(pd.read_csv(DATA_DIR/"Portfolios_Formed_on_AC.csv"))
