from pathlib import Path
from settings import config
import pandas as pd
from datetime import timedelta
from collections import defaultdict
from typing import Union,List,Dict
from pandas.tseries.offsets import BDay
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
import hypothesis_testing
import numpy as np
DATA_DIR = config("DATA_DIR")

code_to_name = { 0:'MISC',1: 'PHASE2_pos', 2: 'PHASE3_pos', 3: 'NEW_DRUG_APPROVAL'}

class Preprocessor:
    def __init__(self, filing_dict_path, return_df_path= None):
        """
        Initialize Preprocessor with filing_dict and return_df
        filing_dict : # dict[dict[list]] {"AAPL":{'text':[],'event':[]...}}
        return_df : index: dates , columns = tickers

        """
        self.filing_dict = pd.read_pickle(filing_dict_path)
        self.return_df = pd.read_parquet(return_df_path) if return_df_path else None

    def sort_events(self,threshold=0):
        """
        Given a filing_dict this method sorts events 
        
        Returns : e.g {'MISC':{'ticker':ticker,'filedAt':filedAt ...},"PHASE1_POS":{'ticker':ticker,'filedAt':filedAt ...}}
        
        You can easily call specific event you want by its key and all the record are in rows
        """
        # Threshold parameter filters out events that were classified with less than (threshold)*100% confidence.

        events = defaultdict(list)

        for ticker, info in self.filing_dict.items():
            for ev, link, filed_at, logprob, text, judge in zip(
                    info.get('event', []),
                    info.get('linkToFilingDetails', []),
                    info.get('filedAt', []),
                    info.get('logprob',[]),
                    info.get('text',[]),
                    info.get('judge_score', [None] * len(info.get('event', [])))):
                
                record = {'ticker': ticker,'filedAt': filed_at,
                    'link': link, 'event': ev,
                    'logprob': max(logprob.values()) if isinstance(logprob, dict) and logprob else None,   # actually a probability no longer logprob
                    'text':text,'judge_score':judge
                }

                # doesn't stack for events below threshold
                if (record['logprob'] is not None) and (record['logprob'] <= threshold): 
                    continue

                # Maps name to the event - this can be declared as a global 
                name = code_to_name.get(ev, 'MISC')
                events[name].append(record) 

        return dict(events) 

    def _event_window(self,events: Dict[str, List[Dict]], prev_window: int = 5, post_window: int = 5) -> pd.DataFrame:
        """
        events: the dict-of-lists output from sort_events() for different events
                
        Returns: a row of -prev_window…+post_window business-day returns,
                 with NaNs filled as 0.

        columns -5 -4 -3 -2 -1 0 1 2 3 4 5
        index: A_2024-11-25, A_2024-09-05 ...
        values: return
        ** All events are flattened out here **
        """
        # flatten sorted events
        if isinstance(events,dict):
            event_df = pd.DataFrame([{**rec} for etype, recs in events.items() for rec in recs])

        elif isinstance(events,list): # 
            event_df = pd.DataFrame(events)

        else:
            raise TypeError(f"`events` must be a dict or list, but got {type(events).__name__}")

        # 2) Parse mixed-offset timestamps into UTC then drop tz, normalize to dates only
        # It drops the time when it's filed, technically subject to a bit of look ahead bias
        # Say the press release was 9 and filed at 16 then it assumes that we get the full day to day return

        event_df['filedAt'] = (
            pd.to_datetime(event_df['filedAt'], utc=True, errors='coerce')
              .dt.normalize()
              .dt.tz_localize(None)
        )

        # 3) Grab wide returns table (one of loaded DataFrames)
        
        wide_df = self.return_df.copy() 

        # Ensure its index is pure dates with no timezone
        wide_df.index = pd.to_datetime(wide_df.index).normalize()

        # 4) Build the event-window
        offsets = list(range(-prev_window, post_window + 1))
        data, index = [], []

        for _, ev in event_df.iterrows(): # iterates through index and series(row)
            tkr   = ev['ticker']
            ev_dt = ev['filedAt']

            # business-day range from ev_dt - set BDays to ev_dt + set BDays
            bdates = pd.bdate_range(
                start= ev_dt - BDay(prev_window),
                end  = ev_dt + BDay(post_window),
                freq ='B'
            )


            if tkr in wide_df.columns:
                rets = (
                    wide_df[tkr]
                      .reindex(bdates,fill_value=0)    # pick exactly those business days
                      .values # fill any gaps with 0
                )
            else:
                # if ticker’s missing entirely, just a zero‐vector 
                rets = [0.0] * len(bdates)

            index.append(f"{tkr}_{ev_dt.date()}") # e.g A_2024-11-25
            data.append(rets)

        return pd.DataFrame(data, index=index, columns=offsets)

    def cumulative_event_window(self,
                                events: Dict[str, List[Dict]],
                                prev_window: int = 5,
                                post_window: int = 5) -> pd.DataFrame:
        """
        ** Takes it only a single event **
        This takes in each event and construct a cumulative ret
        Input events must look like {"FDA_APPROVAL":{'ticker':...}}
        So basically each events sort by events wrapped in dict 

        1) Pull out the raw -prev_window ~ post_window business-day returns with event_window()
        2) Turn them into arithmetic / cumulative returns per event, anchored to 0 at day 0.
        Note: Cumulative returns are mainly for plotting 
        
        """
        # step 1: get the raw window
        ew = self._event_window(events, prev_window, post_window) 

        # step 2: compute cumulative product of (1+ret), subtract 1 to get cumulative returns
        cum_ew = (1 + ew).cumprod(axis=1) - 1

        # step 3: subtract each row’s day-0 value so that at column 0 the cum. return is exactly zero
        cum_ew = cum_ew.sub(cum_ew[0], axis=0)

        # step 4: (optional) fill any remaining NaNs with 0
        return cum_ew.fillna(0)
    
        
    def raw_event_window(self,
                     events: Dict[str, List[Dict]],
                     prev_window: int = 5,
                     post_window: int = 5) -> Dict[str, pd.DataFrame]:
        
        """
        events: sort_events() output
        {'MISC': [{'ticker': 'A','filedAt': '2024-11-25T09:10:09-05:00',
   'link': 'https://www.sec.gov/Archives/edgar/data/1090872/000095017024130441/a-20241125.htm',
   'event': 0,'logprob': 1.0,'text': None,'judge_score': None},..'PHASE1':[....]}


        Returns:
        {'FDA_APPROVAL':df,'PHASE2':df}
            Dict[str, pd.DataFrame] with:
            - index: ticker (deduplicated as A, A_1, A_2, ...)
            - columns: relative day (-prev_window to +post_window)
            - values: raw returns (not cumulative)
        """
        raw_df = self._event_window(events, prev_window, post_window) # flattened out event 

        # Create mapping from event row name (e.g., A_2024-11-25) to event type
        index_to_event = {}
        for phase, recs in events.items():
            for rec in recs:
                tkr = rec['ticker']
                dt = pd.to_datetime(rec['filedAt'], errors='coerce')
                if pd.isna(dt): continue
                key = f"{tkr}_{dt.date()}"
                index_to_event[key] = phase 

        # Assign event_type to each row in raw_df
        raw_df['event_type'] = raw_df.index.map(index_to_event)

        # -3 -2 -1 0 1 2 3 event_type
        #A_2024-04-02 returns and 3

        # Split by event_type, deduplicate tickers
        # result = {"phase_2": DataFrame(...),  # index: AAPL, AAPL_1, MSFT ...
        # "approval": DataFrame(...)  # index: AAPL, AAPL_1 ...}
        
        result = {}
        for phase, group in raw_df.groupby('event_type',sort=True): 
            df = group.drop(columns='event_type')
            ticker_counts = {}
            new_index = []

            for row_id in df.index:
                tkr = row_id.split("_")[0]
                # If first ticker in event A_1
                if tkr not in ticker_counts:
                    new_index.append(tkr)
                    ticker_counts[tkr] = 1
                # else A_2,A_3...
                else:
                    new_index.append(f"{tkr}_{ticker_counts[tkr]}")
                    ticker_counts[tkr] += 1

            df.index = new_index
            result[phase] = df

        return result
    
    
    def _filter_by_lookback(self,events,start_date,lookback_years):
        # This takes in string format date and parse events that occured in between
        start = datetime.fromisoformat(start_date)
        end = start + relativedelta(years=lookback_years)
        to_date = lambda s: datetime.fromisoformat(s[:19])
        return {
            k: [rec for rec in v if start <= to_date(rec['filedAt']) <= end]
            for k, v in events.items()
        }
    def sort_events_walkforward_pvalue(self,lag,max_window,years,events,lookback):
        
        dates = [f"{year}-01-01" for year in years]
        windows = range(lag+1, max_window+1)

        aggregate_dict = {}

        event_types = None
        pos_matrix_by_year = {}

        for idx, year in enumerate(years):
            start_date = f"{year}-01-01"
            # Filtering lookback period
            filtered_events = self._filter_by_lookback(events, start_date=start_date, lookback_years=lookback)
            # Constructing raw with date in between
            raw = self.raw_event_window(filtered_events, prev_window=250, post_window=60)
            
            pos_matrix = pd.DataFrame()
            for t in windows:
                result = hypothesis_testing.t_test(raw, window_t1=lag, window_t2=t, scaled=False)
                pos_values = {evt: result[evt].get("one_sided_positive_pval", np.nan) for evt in result}
                pos_matrix[t] = pd.Series(pos_values)
            pos_matrix_by_year[dates[idx]] = pos_matrix
            if event_types is None:
                event_types = list(pos_matrix.index)

        for event in event_types:
            df = pd.DataFrame(index=dates, columns=windows)
            for date in dates:
                if event in pos_matrix_by_year[date].index:
                    df.loc[date] = pos_matrix_by_year[date].loc[event]
            aggregate_dict[event] = df
        return aggregate_dict
if __name__=="__main__":
    DATA_DIR = config("DATA_DIR")

    
