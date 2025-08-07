import abc
import pandas as pd
from typing import Any, Dict

class Strategy(abc.ABC):
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize strategy with parameter dictionary.
        """
        self.params = params

    @abc.abstractmethod
    def generate_signals(self, return_df: pd.DataFrame) -> pd.DataFrame:
        """
        Given price DataFrame (dates x tickers), return target weights for each date/ticker.

        Implementation steps:
        1. Initialize a signals DataFrame of zeros matching price_df shape.
        2. Loop through your event schedule (e.g., entry/exit dates in params).
        3. At each entry_date, set signals.loc[entry_date, ticker] to initial weight.
        4. At tranche upgrade dates, increase weight accordingly.
        5. After exit_date (or stop-loss), reset weight to 0.

        Returns:
            signals (pd.DataFrame): index = price_df.index, columns = price_df.columns
        """
        # 1) initialize empty signals
        signals = pd.DataFrame(0.0, index=return_df.index, columns=return_df.columns)
        
        # 2) fetch schedule from self.params
        entry_df = self.params.get('entry_dates')      # DataFrame with columns ['ticker','entry_date']
        exit_df  = self.params.get('exit_dates')       # DataFrame with columns ['ticker','exit_date']
        
        # 3) fill in weights
        # for illustration, simple long-only: set full weight between entry and exit
        for _, ev in entry_df.iterrows():
            ticker = ev['ticker']
            start  = ev['entry_date']
            end    = exit_df.loc[exit_df['ticker']==ticker, 'exit_date'].values[0]
            signals.loc[start:end, ticker] = 1.0

        return signals