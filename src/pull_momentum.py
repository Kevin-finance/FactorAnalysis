import pandas as pd
import requests
from io import StringIO
import io
import functools 
import wrds
import numpy as np
import os
import zipfile
from settings import config 

# salrego
# phjwQbTPr.25ZLR


FAMA_DATA_DIR = config("OUTPUT_DIR")
DATA_DIR = config('DATA_DIR')
START_DATE = config("START_DATE")
END_DATE = config("END_DATE")
#WRDS_USERNAME = config("WRDS_USERNAME")
NAME = 'aqr_momentum_series.parquet'


def pull_momentum_data(START_DATE, END_DATE):
    query = f"""
        SELECT
            contrib_global_factor.global_factor.obs_main,
            contrib_global_factor.global_factor.exch_main,
            contrib_global_factor.global_factor.common,
            contrib_global_factor.global_factor.primary_sec,
            contrib_global_factor.global_factor.permno,
            contrib_global_factor.global_factor.date,
            contrib_global_factor.global_factor.shares,
            contrib_global_factor.global_factor.prc,
            contrib_global_factor.global_factor.ret_3_1,
            contrib_global_factor.global_factor.ret_6_1,
            contrib_global_factor.global_factor.ret_9_1,
            contrib_global_factor.global_factor.ret_12_1,
            contrib_global_factor.global_factor.ret_12_7,
            contrib_global_factor.global_factor.resff3_12_1,
            contrib_global_factor.global_factor.resff3_6_1,
            contrib_global_factor.global_factor.ret_2_0,
            contrib_global_factor.global_factor.ret_3_0,
            contrib_global_factor.global_factor.ret_6_0,
            contrib_global_factor.global_factor.ret_9_0,
            contrib_global_factor.global_factor.ret_12_0,
            contrib_global_factor.global_factor.ret_18_1,
            contrib_global_factor.global_factor.ret_24_1,
            contrib_global_factor.global_factor.ret_24_12,
            contrib_global_factor.global_factor.ret_36_1,
            contrib_global_factor.global_factor.ret_36_12,
            contrib_global_factor.global_factor.ret_48_12,
            contrib_global_factor.global_factor.ret_48_1,
            contrib_global_factor.global_factor.ret_60_1,
            contrib_global_factor.global_factor.ret_60_36,
            contrib_global_factor.global_factor.ni_me
        
        FROM contrib_global_factor.global_factor
        
        WHERE contrib_global_factor.global_factor.date BETWEEN
            '{START_DATE}'::date AND '{END_DATE}'::date
        AND contrib_global_factor.global_factor.permno IN (
            '89995'
        )
    """

    query2 = '''SELECT
    comp_na_daily_all.secm.gvkey,
    comp_na_daily_all.secm.datadate,
    comp_na_daily_all.secm.tic,
    comp_na_daily_all.secm.prccm,
    comp_na_daily_all.secm.prchm,
    comp_na_daily_all.secm.prclm,
    comp_na_daily_all.secm.trfm,
    comp_na_daily_all.secm.trt1m,
    comp_na_daily_all.secm.cmth,
    comp_na_daily_all.secm.cyear,
    comp_na_daily_all.secm.cshtrm

    FROM comp_na_daily_all.secm
    INNER JOIN (
        SELECT
            gvkey
        FROM comp_na_daily_all.company
        WHERE comp_na_daily_all.company.gvkey IN (
            '165240'
        )
    ) AS id_table 

    ON comp_na_daily_all.secm.gvkey = id_table.gvkey

    WHERE comp_na_daily_all.secm.datadate BETWEEN
        '2010-01-01'::date AND '2025-07-31'::date'''
    
    query3 = '''SELECT
        comp_na_daily_all.secd.gvkey,
        comp_na_daily_all.secd.datadate,
        comp_na_daily_all.secd.tic,
        comp_na_daily_all.secd.cshoc,
        comp_na_daily_all.secd.cshtrd,
        comp_na_daily_all.secd.prccd,
        comp_na_daily_all.secd.prchd,
        comp_na_daily_all.secd.prcld,
        comp_na_daily_all.secd.prcod

    FROM comp_na_daily_all.secd
    INNER JOIN (
        SELECT
            gvkey
        FROM comp_na_daily_all.company
        WHERE comp_na_daily_all.company.gvkey IN (
            '165240'
        )
    ) AS id_table 

    ON comp_na_daily_all.secd.gvkey = id_table.gvkey

    WHERE comp_na_daily_all.secd.datadate BETWEEN
        '2010-01-01'::date AND '2025-07-02'::date
    '''

    
    db = wrds.Connection()
    momentum_factors = db.raw_sql(query, date_cols=['date']).rename(columns = {'date':'Date'})
    vht_monthly = db.raw_sql(query2, date_cols = ['date']).rename(columns = {'datadate':'Date'})
    vht_daily = db.raw_sql(query3, date_cols = ['date']).rename(columns = {'datadate':'Date'})
    db.close()
    return momentum_factors, vht_monthly, vht_daily



def technical_mom_indicators(monthly:pd.DataFrame,daily:pd.DataFrame):
    monthly['Date'] = pd.to_datetime(monthly['Date'])
    monthly = monthly.set_index('Date')
    daily['Date'] = pd.to_datetime(daily['Date'])
    daily = daily.set_index('Date')
    monthly['log_price'], daily['log_price'] = np.log(monthly['prccm']), np.log(daily['prccd'])
    monthly['return'], daily['return'] = monthly['prccm'].diff(), daily['prccd'].diff()
    log_price = monthly['log_price']
    price, price_d = monthly['prccm'], daily['prccd']
    volume, volume_d = monthly['cshtrm'], daily['cshtrd']
    velocity = log_price.diff()


        # Physics Momentum Definitions. Mass Candidates: volume, total transaction value, and inverse of volatility. Position: log(S(t))
    p_0 = velocity.rolling(window = 30, center = False).sum()
    p_1 = volume*velocity.rolling(window = 30, center = False).sum()
    p_2 = p_1/volume.rolling(window = 30, center = False).sum()
    p_3 = velocity.rolling(window = 30, center = False).mean()/velocity.rolling(window = 30, center = False).std()

        # 30, 200 day exponential moving average
    alpha = 0.92 
    weights = [] 
    ema_30 = daily['prccd'].ewm(span=30, adjust=False).mean().resample('ME').mean()
    ema_200 = daily['prccd'].ewm(span=200, adjust=False).mean().resample('ME').mean()
    ### HERE, We want to use the EMA values, BUT, there is not enough monthly, we will have to get DAILY monthly as well and then use that to calcualte these series. 


    # Relative Strength Indicators, calculating gains and losses
    monthly['gain'] = monthly['return'].clip(lower = 0)
    monthly['loss'] = monthly['return'].clip(upper = 0).abs()
    avg_gain = monthly['gain'].rolling(window = 30, min_periods = 30, center = False).mean()
    avg_loss = monthly['loss'].rolling(window = 30, min_periods = 30, center = False).mean()
    for i,row in enumerate(avg_gain.iloc[31:]):
        avg_gain.iloc[i + 30 + 1] =\
        (avg_gain.iloc[i + 30] *
        (30 - 1) +
        monthly['gain'].iloc[i + 30 + 1])\
        / 30

    for i,row in enumerate(avg_loss.iloc[31:]):
        avg_loss.iloc[i + 30 + 1] =\
        (avg_loss.iloc[i + 30] *
        (30 - 1) +
        monthly['loss'].iloc[i + 30 + 1])\
        / 30

    rs = avg_gain/avg_loss
    rsi = (100-100/(1+rs))

        # Earnings Growth Rate
        # Calculate Earnings per share outstanding. Then just do pct_change()
    # monthly['earnings_per_share']=monthly['ni_me']*price 
    # egr = monthly['earnings_per_share'].pct_change()


        # Commodity Channel Index (CCI)
    def mad(x):
        return np.mean(np.abs(x-np.mean(x)))
        
    high = monthly['prchm']
    low = monthly['prclm']
    typical_price = (high+low+price)/3
    cci = (typical_price - typical_price.rolling(20).mean())/(0.15*typical_price.rolling(20).apply(mad))


        # Average Directional Index (ADX), THIS HAS AN ERROR. LOOK AT THE TRUE RANGE CALCULATION. 
    true_ranges = pd.Series(np.abs(price.diff()))
        # Refer back to avg_gain, avg_loss
    avg_true = true_ranges.rolling(window = 30, center = False, min_periods = 30).mean()
    dx = pd.Series(np.abs((avg_gain-avg_loss)/(avg_gain+avg_loss))*100)
    adx = dx.rolling(window = 30, center = False).mean()


        # MACD
        # Subtract 26 period exponential moving average from 12 period exponential moving average
    ema_26 = price.ewm(span=26, adjust=False).mean()
    ema_12 = price.ewm(span=12, adjust=False).mean()
    macd = ema_12-ema_26
    d = {'ema_30':ema_30, 'ema_200':ema_200, 'p_0':p_0, 'p_1':p_1,'p_2':p_2,'p_3':p_3, 'rsi':rsi, 'cci':cci, 'adx':adx, 'macd':macd}#'egr':egr
    for i,val in d.items():
        d[i] = val.loc['2015-01-01':'2025-06-01']
    output = pd.DataFrame(data = d, index = pd.to_datetime(d['p_0'].index))
    return output


def aqr_ff(folder = DATA_DIR):
    # Here, just need to connect the Fama French api for momentum data, and then to the aqr
    # Currently just have the excel file for AQR Factors. 
    url = r'https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/Momentum-Indices-Monthly.xlsx'
    
    if not os.path.exists(folder):
        os.makedirs(folder)

    try:
        # Download file content
        data = pd.read_excel(url, sheet_name= 'Returns').drop(columns = ['Unnamed: 4','Unnamed: 5',	'AQR Momentum Indices Returns.1','Unnamed: 7','Unnamed: 8']).dropna()
        data.columns = data.iloc[0]
        data = data.drop(index = 0)
        data['Month'] = pd.to_datetime(data['Month'])

        url2 = 'http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Developed_Mom_Factor_CSV.zip'
        res = requests.get(url2)
        res.raise_for_status()

        # Unzip and read the CSV
        with zipfile.ZipFile(io.BytesIO(res.content)) as z:
            file_name = z.namelist()[0]  # usually the only file inside
            with z.open(file_name) as f:
                df = pd.read_csv(f, skiprows=range(0, 6))

        df = df.rename(columns={df.columns[0]: 'Date', df.columns[1]: 'UMD'})
        df = df[df['Date'].str.strip().ne('')].dropna().iloc[:415]
        df['Date'] = df['Date'].str[0:4] + '-' + df['Date'].str[4:]
        df['Date'] = pd.to_datetime(df['Date'], format='mixed')
        df['Date'] = df['Date'] - pd.offsets.DateOffset(months = 1) + pd.offsets.MonthEnd()
        df['Date'] = pd.to_datetime(df['Date'])
        df['UMD'] = pd.to_numeric(df['UMD'], errors='coerce') / 100
        combined = pd.merge(data,df,how = 'inner',right_on=['Date'], left_on = ['Month']).drop(columns = ['Month']).set_index('Date').loc['2015-01-01':'2025-06-01']

    except Exception as e:
        print(f"Failed to process 'AQR/FF': {e}")
    return combined


if __name__ == '__main__':
    momentum_factors, vht_monthly, vht_daily = pull_momentum_data(START_DATE, END_DATE)
    momentum_factors['Date'] = pd.to_datetime(momentum_factors['Date'])
    momentum_factors = momentum_factors.set_index('Date')
    technical = technical_mom_indicators(vht_monthly, vht_daily)
    aqrff = aqr_ff()
    aqrt = pd.merge(technical, aqrff, 'inner',left_index= True, right_index = True)
    output = pd.merge(aqrt,momentum_factors,'inner',right_index = True,left_index = True)

    output.to_parquet(DATA_DIR/"momentum_factors.parquet")