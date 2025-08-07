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

# phjwQbTPr.25ZLR
def pull_esg(n, years = range(2020, 2025),cache_returns_path: Path = None) -> pd.DataFrame:
    """
    Build ESG long-short factor indices:
      - For each metric (fieldname) and each year, pick top-n percent (long) and bottom-n percent (short) tickers by valuescore.
      - Compute equal-weight L/S returns for that year and metric.
    Returns a DataFrame indexed by date with one column per ESG metric.
    """
    if cache_returns_path is None:
        cache_returns_path = DATA_DIR / 'daily_returns.parquet'

    warnings.filterwarnings('ignore')

    tickers = list(pd.read_csv(DATA_DIR / 'russell_2000_tickers.csv')['AAMI'])

    db = wrds.Connection(wrds_username=WRDS_USERNAME)
    try:

        in_list = ','.join([f"'{t}'" for t in tickers])
        esg_query = f"""
            SELECT
                year,
                ticker,
                fieldid,
                hierarchy,
                pillar,
                fieldname,
                valuedate,
                value,
                valuescore
            FROM tresg.wrds_ref_esg
            WHERE ticker IN ({in_list})
              AND year BETWEEN 2020 AND 2025
              AND fieldid IN (
                   1,  2,  3,  4,  5,  6,  7,  8,
                   9, 10, 11, 12, 13, 14, 15, 16
              )
            ORDER BY year, fieldid
        """
        esg = db.raw_sql(esg_query)

        # Here, with annual, I am creating a pivot table that will change our index to the year, and ticker, while making the different esg fields 
        # Column headers, and the values the respective values for each ticker. 
        annual = (
            esg.pivot_table(index=['year', 'ticker'],
                            columns='fieldname',
                            values='valuescore')
                            #aggfunc='mean')
        )
        


        top_dict, bot_dict = {}, {}
        size = max(1,round(len(tickers)*n/100))

        for col in annual.columns:
            s = annual[col].dropna()
            for year, grp in s.groupby(level='year'):
                # drop the 'year' level so index is just 'ticker'
                grp_t = grp.reset_index(level='year', drop=True)
                top_dict[(year, col)] = grp_t.nlargest(size)
                bot_dict[(year, col)] = grp_t.nsmallest(size)

        top_df = (
            pd.concat(top_dict, names=['year', 'metric'])
            .rename('score')              
            .reset_index()                # columns =  year, metric, ticker, score
        )

        bot_df = (
            pd.concat(bot_dict, names=['year', 'metric'])
            .rename('score')
            .reset_index()
)



        # Pairs by rank position
        top_df['pos'] = top_df.groupby(['year','metric']).cumcount()
        bot_df['pos'] = bot_df.groupby(['year','metric']).cumcount()

        pairs = (
            top_df.merge(bot_df,
                        on=['year', 'metric', 'pos'],
                        suffixes=('u', 'd'))
                .drop(columns=['pos', 'scoreu', 'scored'])
                .sort_values(['year', 'metric'])
        )
        all_tickers = sorted(set(pairs['tickeru']) | set(pairs['tickerd']))
        print(len(all_tickers))

        if cache_returns_path.exists():
            returns_df = pd.read_parquet(cache_returns_path)
            returns_df['date'] = pd.to_datetime(returns_df['date'])
            returns_df = returns_df.set_index('date').sort_index()
            returns_df = returns_df.loc[~returns_df.index.duplicated(keep='first'), :]
        else:
            in_list_r = ','.join([f"'{t}'" for t in all_tickers])
            ret_query = f"""
                SELECT dlycaldt AS date, ticker, dlyret
                FROM crsp.dsf_v2
                WHERE dlycaldt BETWEEN '{START_DATE:%Y-%m-%d}' AND '{END_DATE:%Y-%m-%d}'
                AND ticker IN ({in_list_r})
            """
            crsp = db.raw_sql(ret_query)
            crsp['date'] = pd.to_datetime(crsp['date'])

            returns_df = (
                crsp.pivot_table(index='date', columns='ticker', values='dlyret')
                    .sort_index()
            ).drop_duplicates()
            returns_df = returns_df.loc[~returns_df.index.duplicated(keep='first'), :]
            returns_df.to_parquet(cache_returns_path)
        print(returns_df)



        metrics = pairs['metric'].unique()
        index_df = pd.DataFrame(index=returns_df.index, columns=metrics, dtype=float)

        for (year, metric), grp in pairs.groupby(['year', 'metric']):

            mask = returns_df.index.year == year
            long_tickers  = [str(t).strip().upper() for t in grp['tickeru'].tolist()]
            short_tickers = [str(t).strip().upper() for t in grp['tickerd'].tolist()]

            sub = returns_df.loc[mask, :]  # rows for this calendar year

            long_block  = sub.reindex(columns=long_tickers)
            short_block = sub.reindex(columns=short_tickers)

            if long_block.shape[1] == 0 or short_block.shape[1] == 0:
                continue

            long_mean  = long_block.mean(axis=1, skipna = True)   
            short_mean = short_block.mean(axis=1, skipna = True)
            index_df.loc[mask, metric] = long_mean - short_mean

        index_df = index_df.dropna(how='all').drop_duplicates()
        index_df = index_df.groupby(index_df.index).last()

        return index_df

    finally:
        db.close()

def performance_check(n):
    """
    n = Maximum portfolio size to evaluate (will test 10, 20, ..., n).

    best_size = The portfolio size with the lowest sum of ranks.
    rank_sums = Sum of factor‐ranks for each size (index = size).
    ranks_df = The full ranking table (rows = size, cols = factor names).
    """

    sizes = list(range(10, n + 1, 10))

    perf_dict = {}
    for size in sizes:
        returns = pull_esg(size)                  
        cumul   = (returns + 1).cumprod() - 1     
        last    = cumul.iloc[-1].abs()            
        perf_dict[size] = last

    perf_df = pd.DataFrame(perf_dict).T          

    # Here, we rank all of the factors within the dataframe from the smallest to the largest. Then we will go through and sum them to see which had the best overall performance. 

    ranks_df = perf_df.rank(axis=1, ascending=False, method="min")

    rank_sums = ranks_df.sum(axis=1)

    best_size = rank_sums.idxmin()

    best_performing = pull_esg(int(best_size))

    print("Sum of ranks by size:\n", rank_sums, "\n")
    print(f"Best performing portfolio size is {best_size}")

    return best_size, rank_sums, ranks_df, best_performing



if __name__ == '__main__':
    data = pull_esg(10)
    data.to_parquet(DATA_DIR / "ESG_factors.parquet")

    # ADF Test should be conducted twice, first when we have the raw series, and second after we take diff()
    # How to detect if there is an overall trend to the data without doing it manually. 
    # Daily returns for the SQL query, see the other jupyter notebook. 
    # Refer to MSA notes for lambda parameter tuning. 
    # for the grid search we will evaluate via IC or sharpe. 
