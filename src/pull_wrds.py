import pandas as pd
from pandas.tseries.offsets import MonthEnd, YearEnd

import numpy as np
import wrds

from settings import config
from pathlib import Path

OUTPUT_DIR = config("OUTPUT_DIR")
DATA_DIR = config("DATA_DIR")
WRDS_USERNAME = config("WRDS_USERNAME")
START_DATE = config("START_DATE")
END_DATE = config("END_DATE")
FAMA_DATA_DIR  = config('FAMA_DATA_DIR')


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
    fffct_5plusmom.to_csv(FAMA_DATA_DIR/"famafrench_5fct_momentum_monthly.csv")
