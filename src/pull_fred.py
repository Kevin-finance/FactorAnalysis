from settings import config
from fredapi import Fred
import pandas as pd

FRED_API = config("FRED_API")
START_DATE = config("START_DATE")
END_DATE = config("END_DATE")
DATA_DIR = config("DATA_DIR")


fred = Fred(api_key= FRED_API)

# Leave out seasonally adjusted series but for non-seasonally adjusted series, 
# Take diff or log diff to get rid of the trend and see if its there's any ACF

def pull_fred(num,order_by,sort_order,release = 'latest', *args,**kwargs):
    macro_dict = {}
    
    for key,value in kwargs.items():
        df= fred.search_by_category(value,order_by= order_by, sort_order=sort_order)[:num]
        macro_dict[key] = df

    mapping_df = pd.DataFrame(columns=['id','title','frequency_short','seasonal_adjustment_short','units','units_short'])
    
    if release == 'lastest':
        container = []
        column_name = []
        for key,value in macro_dict.items():
            for i in range(len(macro_dict[key]['id'])):
                row = value.iloc[i][['id','title','frequency_short','seasonal_adjustment_short','units','units_short']]
                mapping_df.loc[len(mapping_df)] = row.values

                series = fred.get_series(macro_dict[key]['id'][i]) 
                container.append(series) 
                column_name.append(macro_dict[key]['id'][i])
        
        merged = pd.concat(container,axis=1)
        merged.columns = column_name
        merged = merged.loc[START_DATE:END_DATE]
        merged.dropna(inplace=True, how='all')

    # else: # Need to add asof join pulls
    # Loop through the dates so that we obtain that is available at that time.
    # We will have a time series to be used for each dates.
    # https://github.com/mortada/fredapi/blob/master/fredapi/fred.py
        
    


    return (mapping_df,merged)

if __name__=="__main__":
    # If it has subcategories then it will return no series
    category_id = {"CPI & PCE Prices": 9, "Producer Price Indexes":31,
                   "JOLTS":32241,"Weekly Initial Claims":32240,
                   "GDP": 106, "Unemployment Rate":32247, "Retail Sales":6
                   }
    num = 10 # number of indicators per each categories
    order_by = "popularity" 
    sort_order = "desc"
    mapping_df, merged= pull_fred(num,order_by,sort_order,release = 'lastest',**category_id)
    # monthly data is represented by the first day of the month
    mapping_df.to_parquet(DATA_DIR/"macro_map.parquet")
    merged.to_parquet(DATA_DIR/"macro_latest_series.parquet")

  



