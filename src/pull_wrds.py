import pandas as pd

import numpy as np
import wrds

from settings import config
from pathlib import Path
import pickle

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
    if monthly:
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
    ### pulling holdings of etf
    holdings = pull_lseg(START_DATE, END_DATE , wrds_username=WRDS_USERNAME)
    holdings.to_parquet(DATA_DIR / "vht_holdings.parquet")
    holdings = load_lseg()

    ### pulling famafrench 5 fcts + momentum
    fffct_5plusmom = pull_famafrench_5fctplusmom_monthly_wrds(START_DATE, END_DATE, wrds_username = WRDS_USERNAME)
    fffct_5plusmom.to_csv(DATA_DIR/"famafrench_5fct_momentum_monthly.csv")
    
    ### pulling returns of our interest 
    with open(DATA_DIR/"filings_dict.pkl", "rb") as f:
        filings_dict = pickle.load(f)
        
    tickers = list(filings_dict.keys())
    mon_ret = pull_crsp_returns(START_DATE, END_DATE , tickers = tickers,wrds_username = WRDS_USERNAME, monthly = True)
    mon_ret_df = pd.pivot_table(mon_ret,index='mthcaldt',values='mthret',columns='ticker')
    mon_ret_df.to_parquet(DATA_DIR/"vht_mon_ret.parquet")

    dly_ret = pull_crsp_returns(START_DATE, END_DATE , tickers = tickers,wrds_username = WRDS_USERNAME, monthly = False)
    dly_ret_df = pd.pivot_table(dly_ret,index='dlycaldt',values='dlyret',columns='ticker')
    dly_ret_df.to_parquet(DATA_DIR/"vht_dly_ret.parquet")

    
    ### Code ends

