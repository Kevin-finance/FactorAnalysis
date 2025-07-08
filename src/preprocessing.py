from pathlib import Path
from settings import config
import pandas as pd
from datetime import timedelta
from collections import defaultdict
from typing import Union,List,Dict
from pandas.tseries.offsets import BDay
import matplotlib.pyplot as plt
import itertools

DATA_DIR = config("DATA_DIR")

def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def read_pickle(path: Path):
    # Load pickle, return raw object (DataFrame or dict)
    obj = pd.read_pickle(path)
    return obj

# Mapping file suffixes to reader functions
READERS = {
    '.parquet': read_parquet,
    '.pkl':   read_pickle,
    '.pickle': read_pickle,
}


class Preprocessor:
    def __init__(self, *file_paths):
        """
        Initialize Preprocessor with one or more data files.
        .parquet/.pickle DataFrames are stored separately in self.dataframes;
        .pkl/.pickle dict objects are stored in self.filing_dict.
        """
        self.dataframes = {}          # map filename to DataFrame
        self.filing_dict = None       # raw dict from pickle
        self._load_files(file_paths)

    def _load_files(self, file_paths):
        for p in file_paths:
            path = Path(p)
            ext = path.suffix.lower()
            reader = READERS.get(ext)
            if reader is None:
                raise ValueError(f"Unsupported file type: {ext}")

            obj = reader(path)
            if isinstance(obj, pd.DataFrame):
                # store each DataFrame separately
                self.dataframes[path.name] = obj.copy()
            
            elif isinstance(obj, dict):
                # store raw filing dictionary
                self.filing_dict = obj.copy()
            else:
                raise TypeError(f"Unsupported object in file '{path.name}': {type(obj)}")

    def get_dataframe(self, name: str) -> pd.DataFrame:
        """
        Return a copy of the DataFrame loaded from file 'name'.
        """
        if name not in self.dataframes:
            raise KeyError(f"DataFrame '{name}' not found. Available: {list(self.dataframes.keys())}")
        return self.dataframes[name].copy()

    def list_dataframes(self) -> list:
        """
        List names of all loaded DataFrame files.
        """
        return list(self.dataframes.keys())

    def get_filing_dict(self) -> dict:
        """
        Return the raw filing dictionary loaded from pickle.
        """
        if self.filing_dict is None:
            raise ValueError("No filing_dict available. Load a pickle file first.")
        return self.filing_dict.copy()
    
    def sort_events(self):
       
        code_to_name = {
            1: 'PHASE1', 2: 'PHASE2', 3: 'PHASE3',
            4: 'NDA/BLA', 5: 'CRL', 6: 'FDA_APPROVAL'
        }
        events = defaultdict(list)

        for ticker, info in self.filing_dict.items():
            for ev, link, filed_at in zip(
                    info.get('event', []),
                    info.get('linkToFilingDetails', []),
                    info.get('filedAt', [])):
                name = code_to_name.get(ev, 'MISC')
                record = {
                    'ticker': ticker,
                    'filedAt': filed_at,
                    'link': link,
                    'event': ev
                }
                events[name].append(record)

        return dict(events)

    def event_window(self,
                     events: Dict[str, List[Dict]],
                     prev_window: int = 5,
                     post_window: int = 5) -> pd.DataFrame:
        """
        events: either a flat DataFrame with ['ticker','filedAt',...] or
                the dict-of-lists output from sort_events()
        Returns: for each event, a row of -prev_window…+post_window business-day returns,
                 with NaNs filled as 0.
        """
        # 1) Flatten the dict-of-lists into a DataFrame 
        if isinstance(events, dict):
            flat_list = list(itertools.chain.from_iterable(events.values()))
            event_df = pd.DataFrame(flat_list)
            ### OLD
            # rows = []
            # for recs in events.values():
            #     rows.extend(recs)
            # event_df = pd.DataFrame(rows)
            ###
        else:
            event_df = events.copy()

        # 2) Parse mixed-offset timestamps into UTC then drop tz, normalize to dates only
        # It drops the time when it's filed, technically subject to a bit of look ahead bias
        # Say the press release was 9 and filed at 16 then it assumes that we get the full day to day return

        event_df['filedAt'] = (
            pd.to_datetime(event_df['filedAt'], utc=True, errors='coerce')
              .dt.normalize()
              .dt.tz_localize(None)
        )

        # 3) Grab wide returns table (one of loaded DataFrames)
        # This assumes the first dataframe is the return table
        wide_df = next(iter(self.dataframes.values())).copy()

        # Ensure its index is pure dates
        wide_df.index = pd.to_datetime(wide_df.index).normalize()

        # 4) Build the event-window
        offsets = list(range(-prev_window, post_window + 1))
        data, index = [], []

        for _, ev in event_df.iterrows(): # iterates through index and series(row)
            tkr   = ev['ticker']
            ev_dt = ev['filedAt']

            # business-day range from ev_dt - set BDays to ev_dt + set BDays
            bdates = pd.bdate_range(
                start=ev_dt - BDay(prev_window),
                end  =ev_dt + BDay(post_window),
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

            index.append(f"{tkr}_{ev_dt.date()}") # e.g A_datetime(2024,11,25)
            data.append(rets)

        return pd.DataFrame(data, index=index, columns=offsets)

    def cumulative_event_window(self,
                                events: Dict[str, List[Dict]],
                                prev_window: int = 5,
                                post_window: int = 5) -> pd.DataFrame:
        """
        1) Pull out the raw -prev_window ~ post_window business-day returns with event_window()
        2) Turn them into arithmetic / cumulative returns per event, anchored to 0 at day 0.
        Note: Cumulative returns are mainly for plotting 
        
        """
        # step 1: get the raw window
        ew = self.event_window(events, prev_window, post_window)

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
        # Need to Review
        """
        Returns:
            Dict[str, pd.DataFrame] with:
            - index: ticker (deduplicated as A, A_1, A_2, ...)
            - columns: relative day (-prev_window to +post_window)
            - values: raw returns (not cumulative)
        """
        raw_df = self.event_window(events, prev_window, post_window)

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

        # Split by event_type, deduplicate tickers
        result = {}
        for phase, group in raw_df.groupby('event_type'):
            df = group.drop(columns='event_type')
            ticker_counts = {}
            new_index = []

            for row_id in df.index:
                tkr = row_id.split("_")[0]
                if tkr not in ticker_counts:
                    new_index.append(tkr)
                    ticker_counts[tkr] = 1
                else:
                    new_index.append(f"{tkr}_{ticker_counts[tkr]}")
                    ticker_counts[tkr] += 1

            df.index = new_index
            result[phase] = df

        return result


    def get(self) -> pd.DataFrame:
        """
        """
        return self.dataframes


if __name__=="__main__":
    filings = DATA_DIR / "filings_dict.pkl"
    dly_ret = DATA_DIR / "vht_dly_ret.parquet"

