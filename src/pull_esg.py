import pandas as pd
import requests
from io import StringIO
import functools 
import warnings
from pathlib import Path
import wrds
import matplotlib.pyplot as plt
import os
from settings import config 


FAMA_DATA_DIR = config("OUTPUT_DIR")
DATA_DIR = config('DATA_DIR')
START_DATE = config("START_DATE")
END_DATE = config("END_DATE")
WRDS_USERNAME = config("WRDS_USERNAME")
tickers = list()

def pull_esg(n):
    # This is a bit more difficult to do, We have the monthly scores of ESG Data but it is company specific, so I will weight the companies by their contribution 
    # to VHT. This will be difficult to do dynamically. Ask about the dynamic data. 
    warnings.filterwarnings('ignore')
    
    tickers = list(pd.read_csv(DATA_DIR / r'russell_2000_tickers.csv')['AAMI'])

    db = wrds.Connection()

    # salrego
    # phjwQbTPr.25ZLR

    output = pd.DataFrame()
    for t in tickers: 
        query = f'''SELECT
        tresg.wrds_ref_esg.year,
        tresg.wrds_ref_esg.ticker,
        tresg.wrds_ref_esg.fieldid,
        tresg.wrds_ref_esg.hierarchy,
        tresg.wrds_ref_esg.pillar,
        tresg.wrds_ref_esg.fieldname,
        tresg.wrds_ref_esg.valuedate,
        tresg.wrds_ref_esg.value,
        tresg.wrds_ref_esg.valuescore
    FROM tresg.wrds_ref_esg
    WHERE
        tresg.wrds_ref_esg.ticker = '{t}'                         -- qvar
    AND tresg.wrds_ref_esg.year   BETWEEN 2020 AND 2025          -- beg_yr / end_yr
    AND tresg.wrds_ref_esg.fieldid IN (                          -- extra2 + extra2_op=in
            1,  2,  3,  4,  5,  6,  7,  8,
            9, 10, 11, 12, 13, 14, 15, 16
    )
    ORDER BY
        tresg.wrds_ref_esg.year,
        tresg.wrds_ref_esg.fieldid;'''

        data = db.raw_sql(query)
        output = pd.concat([output,data])

    new = output.copy()
    pivot = output.pivot_table(index=['year', 'ticker'], columns='fieldname', values='valuescore', aggfunc='mean').reset_index()
    annual = pivot.set_index(['year','ticker'])
    top = {}
    bottom = {}
    years = [2020,2021,2022,2023,2024]

    for col in annual:
            # if col != 'ticker' and col != 'year':
            #     annual[col] = annual[col].sort_values(ignore_index = True)
        temp = pd.Series(annual[col].sort_values(ignore_index = False))
        for year in years:
            top[(year,col)] = temp.loc[year].iloc[-10:]
            bottom[(year,col)] =  temp.loc[year].iloc[0:10]

    # 1. Wide table with MultiIndex (year, ticker)
    annual = (
        output
        .pivot_table(index=['year', 'ticker'],
                    columns='fieldname',
                    values='valuescore',
                    aggfunc='mean')
    )

    years = [2020, 2021, 2022, 2023, 2024]
    top = {}
    bottom = {}
    new_tickers = set()

    for col in annual.columns:                           
        s = annual[col].dropna()                         
        
        # group by year
        grouped = s.groupby(level='year')
        
        for year in years:
            if year in grouped.groups:               
                top[(year, col)]    = grouped.get_group(year).nlargest(n)
                bottom[(year, col)] = grouped.get_group(year).nsmallest(n)

    top_df = (
        pd.concat(top, names=['year', 'metric'])
        .reset_index(level=2)             
        .rename(columns={'level_2': 'ticker', 0: 'score'})
    )
    top_df['side'] = 1

    bottom_df = (
        pd.concat(bottom, names=['year', 'metric'])
        .reset_index(level=2)
        .rename(columns={'level_2': 'ticker', 0: 'score'})
    )
    bottom_df['side'] = -1

    tickers_top    = set(top_df.index.get_level_values('ticker'))
    tickers_bottom = set(bottom_df.index.get_level_values('ticker'))
    merged = pd.concat([top_df, bottom_df])

    all_tickers = list(tickers_top | tickers_bottom)
    
    daily_returns_path = DATA_DIR / 'daily_returns.parquet'

    top_df   = top_df.copy()
    bottom_df = bottom_df.copy()

    top_df['pos']    = top_df.groupby(level=['year','metric']).cumcount() 
    bottom_df['pos'] = bottom_df.groupby(level=['year','metric']).cumcount()

    top_wide    = top_df.drop(columns = ['year']).reset_index(level = [0,1,2])
    bottom_wide = bottom_df.drop(columns = ['year']).reset_index(level = [0,1,2])

    paired = (
            top_wide
            .merge(bottom_wide,
                    on=['year', 'metric', 'pos'],     
                    suffixes=('u', 'd'))            
        )

        # tidy up (optional)
    paired = (paired
                    .drop(columns=['pos'])             
                    .sort_values(['year', 'metric']))

    metrics = set(paired['metric'])

    if daily_returns_path.exists():
        new = pd.read_parquet(daily_returns_path)

    else:
        stock_data = {}
        for ticker in all_tickers:
            try:
                query2 = f"""
                                    Select a.permno, a.permco, a.dlycaldt, a.ticker,
                                    a.issuertype, a.securitytype, a.securitysubtype, a.sharetype, a.usincflg, 
                                    a.primaryexch, a.conditionaltype, a.tradingstatusflg,
                                    a.dlyret, a.dlyretx, a.shrout, a.dlyprc
                                    from crsp.dsf_v2 as a
                                    where a.dlycaldt between '2020-01-01' and '2025-07-01'
                                    AND a.ticker ='{ticker}'
                                """
                data = db.raw_sql(query2)
                stock_data[ticker] = data.sort_values(by = 'dlycaldt')
                print(ticker)
            except:
                print(f'Data not found for {ticker}')
        
        new = None                                        
        for tic, df in stock_data.items():
            if df is None or df.empty:
                continue
            concat = (df[['dlycaldt', 'dlyret']].rename(columns={'dlycaldt': 'date', 'dlyret': tic}))
            concat['date'] = pd.to_datetime(concat['date'])

            if new is None:
                new = concat                               
            else:
                if tic not in new.columns:
                    new = new.merge(concat, how='outer', on='date').fillna(0)
    def create_indices(weekly_returns, esg_pairs):
        
        index_df = weekly_returns.copy()
        
        # Convert date column to datetime if it's not already
        index_df['date'] = pd.to_datetime(index_df['date'])
        index_df['year'] = index_df['date'].dt.year
        
        # Get unique metrics
        unique_metrics = esg_pairs['metric'].unique()
        
        # Initialize strategy columns
        for metric in unique_metrics:
            index_df[f'{metric}'] = 0.0
        
        # Process each year and metric combination
        for year in esg_pairs['year'].unique():
            # Filter weekly returns for this year
            year_mask = index_df['year'] == year
            
            for metric in unique_metrics:
                # Get the long/short pairs for this year and metric
                pairs = esg_pairs[(esg_pairs['year'] == year) & (esg_pairs['metric'] == metric)]
                
                if len(pairs) == 0:
                    continue
                
                # Calculate strategy returns for each row in this year
                for idx in index_df[year_mask].index:
                    upper_return = 0
                    lower_return = 0
                    valid_u = 0
                    valid_d = 0
                    
                    for _, pair in pairs.iterrows():
                        long_ticker = pair['tickeru']
                        short_ticker = pair['tickerd']
                        
                        # Check if both tickers exist in the dataframe
                        if long_ticker in index_df.columns and short_ticker in index_df.columns:
                            long_ret = index_df.loc[idx, long_ticker]
                            short_ret = index_df.loc[idx, short_ticker]
                            
                            if pd.notna(long_ret) and pd.notna(short_ret):
                                upper_return += long_ret 
                                lower_return += short_ret
                                valid_d += 1
                                valid_u += 1
                    # Take the average over all upper and lower separately. Then long the upper and short the lower. 
                    if valid_d > 0 and valid_u > 0:
                        index_df.loc[idx, f'{metric}'] = (upper_return / n) - (lower_return/n)
        
        # Remove the temporary year column
        index_cols = [col for col in index_df.columns if col.startswith('ESG')]
        index_df = index_df[['date'] + index_cols].set_index('date')
        index_df = index_df[~index_df.index.duplicated(keep='first')]
        
        return index_df

    final = create_indices(new, paired)
    return final


if __name__ == '__main__':
    data = pull_esg(10)
    data.to_parquet(DATA_DIR/"ESG_factors.parquet")

    # Notes: With only using ESG scores from 100 of these stocks which already has some selection bias, we will have incredibly biased ESG factors, not truly global. 
    # COnsider expanding the universe to more stocks for this specifically. 
    # ADF Test should be conducted twice, first when we have the raw series, and second after we take diff()
    # How to detect if there is an overall trend to the data without doing it manually. 
    # Daily returns for the SQL query, see the other jupyter notebook. 
    # Refer to MSA notes for lambda parameter tuning. 
    # for the grid search we will evaluate via IC or sharpe. 
