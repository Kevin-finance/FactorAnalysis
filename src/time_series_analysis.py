from statsmodels.tsa.stattools import adfuller
from settings import config
from preprocessing import Preprocessor
import pandas as pd 
import numpy as np
import utils

DATA_DIR = config("DATA_DIR")
OUTPUT_DIR  = config("OUTPUT_DIR")


def adf_test(df,map_df,trend_id,alpha=0.05):
    # Divide series into one that has a trend and one that does not. 
    trend_id = trend_id

    adf_df = pd.DataFrame(columns = ['ADF','pvalue','usedlag','nobs','critical_values','icbest'])
    for id in df.columns.tolist():
        series = df[id].dropna()
        # We should take care of maxlag
        try:
            if id in trend_id:
                adf,pvalue,usedlag, nobs, critical_values,icbest = adfuller(series, regression='ct', maxlag = 1, autolag = 't-stat')
                adf_df.loc[id] = [adf,pvalue,usedlag,nobs,critical_values[f"{int(alpha * 100)}%"],icbest]
            else:
                adf,pvalue,usedlag, nobs, critical_values,icbest = adfuller(series,regression='c', maxlag = 1, autolag = 't-stat')
                adf_df.loc[id] = [adf,pvalue,usedlag,nobs,critical_values[f"{int(alpha * 100)}%"],icbest]
        except:
            pass

    adf_df['null'] = np.where(adf_df['pvalue']<alpha,'Accept','Reject')
    adf_df['desc'] = adf_df.index.map(map_df.set_index('id')['title'])
    adf_df['units_short'] = adf_df.index.map(map_df.set_index('id')['units_short'])
    adf_df = adf_df.sort_values(by='pvalue',ascending=False)
    return adf_df 

def difference(df,adf_df):
    # Difference the series if not stationary. Log differencing for levels and differencing for rates.
    filter_rates_id = adf_df[(adf_df['null'] == 'Reject') & (adf_df['units_short'].str.contains('%'))].index
    # filter_level_id = adf_df[(adf_df['null'] == 'Reject') & (~adf_df['units_short'].str.contains('%'))].index
    
    for id in df.columns:
        df_clean = df[id].dropna()
        if id in filter_rates_id:
            df[id] = df_clean.diff().reindex(df.index)
        else:
            df[id] = np.log(df_clean).diff().reindex(df.index)

    return df

if __name__=="__main__":
    macro_map_dir = DATA_DIR/"macro_map.parquet"
    macro_latest_series = DATA_DIR/"macro_latest_series.parquet"

    macro_map = Preprocessor(macro_map_dir).get()
    macro_latest_series = Preprocessor(macro_latest_series).get()

    trend_id = ['CPIAUCSL','PCEPI','PCEPILFE','CPIAUCNS','CWUR0000SA0','PCECTPI', # CPI / PCE
                "PPIACO","PCUOMFGOMFG","PCUATRNWRATRNWR","PCUARETTRARETTR","PCUAWHLTRAWHLTR","PCUATRANSATRANS","PCUAINFOAINFO",'PCUASHCASHC','PCUADLVWRADLVWR' #PPI
                "GDP","GDPC1","A939RX0Q048SBEA","GDPPOT","A191RL1Q225SBEA","GDPA","GNP","A261RX1Q020SBEA", #GDP
                "RSXFS","RSAFS","ECOMSA","RETAILIMSA","MRTSSM44000USS" # retail sales
                
                ]
    adf_df = adf_test(macro_latest_series,macro_map,trend_id)
    differenced_df = difference(macro_latest_series,adf_df)
   
    fig = utils.plot_raw_series_subplots(differenced_df, macro_map)
    fig.write_html(OUTPUT_DIR/"macro_raw_post_differencing.html")

    fig2 = utils.plot_acf_subplots(differenced_df, macro_map)
    fig2.write_html(OUTPUT_DIR/"macro_acf_post_differencing.html")

    adf_df_post = adf_test(differenced_df,macro_map,trend_id)
    print(adf_df_post)
