import pandas as pd

import numpy as np
import wrds

from settings import config
from pathlib import Path

OUTPUT_DIR = config("OUTPUT_DIR")
DATA_DIR = config("DATA_DIR")
WRDS_USERNAME = config("WRDS_USERNAME")
START_DATE = config("START_DATE")
END_DATE = config("END_DATE")


def pull_lseg(START_DATE, END_DATE,wrds_username = WRDS_USERNAME):
    sql_query = f"""
        SELECT
        tr_mutualfunds.s12.fundno,
        tr_mutualfunds.s12.rdate,
        tr_mutualfunds.s12.fdate,
        tr_mutualfunds.s12.cusip,
        tr_mutualfunds.s12.stkname,
        tr_mutualfunds.s12.ticker

        FROM tr_mutualfunds.s12

        WHERE tr_mutualfunds.s12.rdate BETWEEN
        '{START_DATE}'::date AND '{END_DATE}'::date
        AND tr_mutualfunds.s12.fundno IN (
        '42819'
        )
        """
    db = wrds.Connection(wrds_username=wrds_username)
    holdings = db.raw_sql(sql_query, date_cols=["datadate"])
    db.close()

    return holdings


def fetch_crsp_data(tickers, start_date, end_date, wrds_username):
    """
    Fetches specified stock data from WRDS CRSP table (crsp_a_stock.wrds_dsfv2_query).

    Parameters:
    tickers (list): List of ticker symbols to retrieve.
    start_date (str): Starting date in 'YYYY-MM-DD' format.
    end_date (str): Ending date in 'YYYY-MM-DD' format.
    wrds_username (str): Your WRDS username.

    Returns:
    pd.DataFrame: DataFrame containing requested data.
    """

    db = wrds.Connection(wrds_username=wrds_username)

    query = f"""
        SELECT
            dispermno,            -- PERMNO of the Security Received
            dispermco,            -- PERMCO of the Issuer Providing Payment
            issuertype,           -- Issuer Type
            securitytype,         -- Security Type
            securitysubtype,      -- Security Sub-Type
            dlyret,               -- Daily Total Return
            dlyretx,              -- Daily Price Return
            shrout,               -- Shares Outstanding
            dlyprc,               -- Daily Price
            tradingstatusflg,     -- Trading Status Flag
            usincflg,             -- US Incorporation Flag
            primaryexch,          -- Primary Exchange
            conditionaltype,      -- Conditional Type
            ticker,               -- Ticker
            dlycaldt              -- Daily Calendar Date
        FROM crsp_a_stock.wrds_dsfv2_query
        WHERE ticker IN ({','.join([f"'{t}'" for t in tickers])})
        AND dlycaldt BETWEEN '{start_date}' AND '{end_date}'
    """
    
    df = db.raw_sql(query)
    db.close()

    return df


def pull_famafrench_5fctplusmom_monthly_wrds(start_date, end_date, wrds_username):
    """_summary_

    Args:
        start_date (_type_): start date
        end_date (_type_): end date
        wrds_username (_type_): wrds username
    
    Returns:
        pd.DataFrame: df includes ['date', 'mktrf', 'smb', 'hml', 'rmw', 'cma', 'umd', 'rf']
    """
    
    db = wrds.Connection(wrds_username=wrds_username)
    query = f"""
        SELECT date, mktrf, smb, hml, rmw, cma, umd, rf
        FROM ff_all.fivefactors_monthly
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY date
    """
    df = db.raw_sql(query, date_cols = ['date'])
    db.close()
    
    df.set_index('date', inplace = True)
    df = df.astype(float)
    
    return df


def load_lseg(data_dir=DATA_DIR):
    path = Path(data_dir) / "vht_holdings.parquet"
    comp = pd.read_parquet(path)
    return comp

def _demo():
    holdings = load_lseg(DATA_DIR)


if __name__ == "__main__":
    holdings = pull_lseg(START_DATE, END_DATE , wrds_username=WRDS_USERNAME)
    holdings.to_parquet(DATA_DIR / "vht_holdings.parquet")

    holdings = load_lseg()
    print(holdings)
    
    fffct_5plusmom = pull_famafrench_5fctplusmom_monthly_wrds(START_DATE, END_DATE, wrds_username = WRDS_USERNAME)
    fffct_5plusmom.to_csv(DATA_DIR/"famafrench_5fct_momentum_monthly.csv")
    
    df = holdings.copy()
    df['holdings'] = 1
    ticker_list = df.pivot_table(index = 'rdate', columns = 'ticker', values = 'holdings').columns.to_list()
    daily_rtn = fetch_crsp_data(ticker_list, START_DATE, END_DATE, wrds_username = WRDS_USERNAME)
    daily_rtn.to_parquet(DATA_DIR / "vht_stk_daily_rtn.parquet")
