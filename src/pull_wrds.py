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

def pull_crsp_returns(START_DATE, END_DATE, tickers ,wrds_username = WRDS_USERNAME, monthly= True):

    """Pull necessary returns of basket of VHT holding regardless of its delisted status
    mthret: Monthly return including dividends
    mthretx: Monthly return excluding dividends
    mthprc: 
    
    
    """
    db = wrds.Connection(wrds_username=wrds_username)
    ticker_str = ', '.join([f"'{t}'" for t in tickers])
    if monthly == True:
        sql_query = f"""
                        Select a.permno, a.permco, a.mthcaldt, a.ticker,
                        a.issuertype, a.securitytype, a.securitysubtype, a.sharetype, a.usincflg, 
                        a.primaryexch, a.conditionaltype, a.tradingstatusflg,
                        a.mthret, a.mthretx, a.shrout, a.mthprc
                        from crsp.msf_v2 as a
                        where a.mthcaldt between '{START_DATE}' and '{END_DATE}'
                        AND a.ticker IN ({ticker_str})
                    """
        crsp = db.raw_sql(sql_query, date_cols=["mthcaldt"])
    else: 
        sql_query = f"""
                        Select a.permno, a.permco, a.dlycaldt, a.ticker,
                        a.issuertype, a.securitytype, a.securitysubtype, a.sharetype, a.usincflg, 
                        a.primaryexch, a.conditionaltype, a.tradingstatusflg,
                        a.dlyret, a.dlyretx, a.shrout, a.dlyprc
                        from crsp.dsf_v2 as a
                        where a.dlycaldt between '{START_DATE}' and '{END_DATE}'
                        AND a.ticker IN({ticker_str})
                    """
        crsp = db.raw_sql(sql_query, date_cols=["dlycaldt"])
    db.close()

    return crsp





def load_lseg(data_dir=DATA_DIR):
    path = Path(data_dir) / "vht_holdings.parquet"
    comp = pd.read_parquet(path)
    return comp

def _demo():
    holdings = load_lseg(DATA_DIR)


if __name__ == "__main__":
    # holdings = pull_lseg(START_DATE, END_DATE , wrds_username=WRDS_USERNAME)
    # holdings.to_parquet(DATA_DIR / "vht_holdings.parquet")

    # holdings = load_lseg()
    temp = pull_crsp_returns(START_DATE, END_DATE , tickers = ['AAPL','MSFT'],wrds_username = WRDS_USERNAME, monthly = True)
    print(temp)
