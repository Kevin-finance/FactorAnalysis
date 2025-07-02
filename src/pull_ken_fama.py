import pandas as pd
import requests
from bs4 import BeautifulSoup
from io import StringIO, BytesIO
import os
import zipfile
from settings import config

FAMA_DATA_DIR = config("FAMA_DATA_DIR")
START_DATE =  config("START_DATE")
END_DATE = config("END_DATE")
DATA_LIB_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html"

FACTOR_DESCRIPTIONS = [
    "Portfolios Formed on Operating Profitability [ex. Dividends]",
    "Portfolios Formed on Investment [ex. Dividends]",
    "Portfolios Formed on Earnings/Price [ex. Dividends]",
    "Portfolios Formed on Cashflow/Price [ex. Dividends]",
    "Portfolios Formed on Dividend Yield [ex. Dividends]",
    "Short-Term Reversal Factor (ST Rev)",
    "Long-Term Reversal Factor (LT Rev)",
    "Portfolios Formed on Accruals",
    "Portfolios Formed on Market Beta",
    "Portfolios Formed on Net Share Issues",
    "Portfolios Formed on Variance",
    "Portfolios Formed on Residual Variance"
]

def download_and_process_zip(factor_descriptions, save_folder=FAMA_DATA_DIR):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    res = requests.get(DATA_LIB_URL)
    res.raise_for_status()

    soup = BeautifulSoup(res.content, "html.parser")

    for desc in factor_descriptions:
        link_found = False
        for factor in soup.find_all("b"):
            if desc in factor.text:
                zip_link = factor.find_next("a", string="TXT")
                if zip_link:
                    zip_href = zip_link.get("href")
                    full_url = f"https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/{zip_href}"
                    print(f"Downloading and processing: {desc} from {full_url}")

                    file_res = requests.get(full_url)
                    file_res.raise_for_status()

                    with zipfile.ZipFile(BytesIO(file_res.content)) as z:
                        for txt_file in z.namelist():
                            with z.open(txt_file) as file:
                                data_str = file.read().decode('latin1')
                                df = auto_read_first_table_from_txt(data_str)

                                csv_file_name = os.path.splitext(txt_file)[0] + '.csv'
                                csv_file_path = os.path.join(save_folder, csv_file_name)
                                df.to_csv(csv_file_path)
                                print(f"Processed and saved to {csv_file_path}")

                    link_found = True
                    break

        if not link_found:
            print(f"Could not find ZIP link for '{desc}'.")

def auto_read_first_table_from_txt(data_str):
    lines = data_str.strip().split('\n')

    data_start = next(idx for idx, line in enumerate(lines) if line.strip().startswith(('19', '20')))
    data_end = next((idx for idx in range(data_start, len(lines)) if lines[idx].strip() == ""), len(lines))
    
    data_lines = lines[data_start:data_end]
    header_line = lines[data_start - 1]
    
    table_str = '\n'.join([header_line] + data_lines)
    df = pd.read_fwf(StringIO(table_str), index_col=0)
    df.index = pd.to_datetime(df.index, format='%Y%m', errors='coerce')
    
    df = df[(df.index >= START_DATE) & (df.index <= END_DATE)]
    df.index.name = 'date'

    df.replace([-99.99, -999], pd.NA, inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce') / 100

    return df

if __name__ == "__main__":
    download_and_process_zip(FACTOR_DESCRIPTIONS)
