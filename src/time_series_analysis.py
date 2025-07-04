from statsmodels.tsa.stattools import adfuller
from settings import config
from preprocessing import Preprocessor
import pandas as pd 
import numpy as np
import utils

DATA_DIR = config("DATA_DIR")
OUTPUT_DIR  = config("OUTPUT_DIR")


def adf_test(df,map_df,trend_id = None,alpha=0.05,second_pass = False ):
    # Divide series into one that has a trend and one that does not. 
    # After differencing we should be running adf on regression ='c'
    
    cols = ['ADF','pvalue','usedlag','nobs','critical_values','icbest']

    adf_df = pd.DataFrame(columns = cols)
    for id in df.columns.tolist():
        series = df[id].dropna()
        
        if second_pass:
            reg = 'c'
        else: 
            reg = 'ct' if id in trend_id else 'c' 

        try:
            result = adfuller(series, regression=reg, max_lag=1, autolag = 'tstat')
            adf, pval, lag, nobs, crit_vals, icbest = result
            cv = crit_vals[f"{int(alpha*100)}%"]
            adf_df.loc[id] = [adf,pval,lag,nobs,cv,icbest]
        except Exception:
            continue

    adf_df['null'] = np.where(adf_df['pvalue']<alpha,True,False) # if it's true then its nonstationary
    adf_df['desc'] = adf_df.index.map(map_df.set_index('id')['title'])
    adf_df['units_short'] = adf_df.index.map(map_df.set_index('id')['units_short'])
    adf_df = adf_df.sort_values(by='pvalue',ascending=False)
    return adf_df 

def difference(df,adf_df):
    # Difference the series if not stationary. Log differencing for levels and differencing for rates.
    # If adf test is rejected then we are accepting hypothesis that it is stationary.
    filter_rates_id = adf_df[(adf_df['null']) & (adf_df['units_short'].str.contains('%'))].index
    filter_level_id = adf_df[(adf_df['null']) & (~adf_df['units_short'].str.contains('%'))].index

    for id in df.columns:
        df_clean = df[id].dropna()
        if id in filter_rates_id:
            df[id] = df_clean.diff().reindex(df.index)
        elif id in filter_level_id:
            df[id] = np.log(df_clean).diff().reindex(df.index)
        else: 
            df[id] = df_clean.reindex(df.index)

    return df

if __name__=="__main__":

    trend_id = ['CPIAUCSL','PCEPI','PCEPILFE','CPIAUCNS','CWUR0000SA0','PCECTPI', # CPI / PCE
                "PPIACO","PCUOMFGOMFG","PCUATRNWRATRNWR","PCUARETTRARETTR","PCUAWHLTRAWHLTR","PCUATRANSATRANS","PCUAINFOAINFO",'PCUASHCASHC','PCUADLVWRADLVWR' #PPI
                "GDP","GDPC1","A939RX0Q048SBEA","GDPPOT","A191RL1Q225SBEA","GDPA","GNP","A261RX1Q020SBEA", #GDP
                "RSXFS","RSAFS","ECOMSA","RETAILIMSA","MRTSSM44000USS" # retail sales
                
                ]
    macro_map_dir = DATA_DIR/"macro_map.parquet"
    macro_latest_series = DATA_DIR/"macro_latest_series.parquet"

    macro_map = Preprocessor(macro_map_dir).get()
    macro_latest_series = Preprocessor(macro_latest_series).get()

    # Plotting the raw series
    fig = utils.plot_raw_series_subplots(macro_latest_series, macro_map)
    fig.write_html(OUTPUT_DIR/"macro_raw.html")
    # Plotting the ACF of raw series
    fig2 = utils.plot_acf_subplots(macro_latest_series, macro_map)
    fig2.write_html(OUTPUT_DIR/"macro_acf.html")

    # Running the adf test with raw series, identified the series with trend with eye ball test
    adf_df = adf_test(macro_latest_series,macro_map,trend_id) 

    # Once we have the adf results for raw series, we decide which one for difference and not 
    differenced_df = difference(macro_latest_series,adf_df)
   
    fig = utils.plot_raw_series_subplots(differenced_df, macro_map)
    fig.write_html(OUTPUT_DIR/"macro_raw_post_differencing.html")

    fig2 = utils.plot_acf_subplots(differenced_df, macro_map)
    fig2.write_html(OUTPUT_DIR/"macro_acf_post_differencing.html")

    adf_df_post = adf_test(differenced_df,macro_map,second_pass=True)
    print(adf_df_post)
