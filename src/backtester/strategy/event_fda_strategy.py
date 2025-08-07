import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent  # FactorAnalysis/src 의 상위
sys.path.insert(0, str(root))

from settings import config
from preprocessing import Preprocessor
from backtester.strategy.base import Strategy
import pandas as pd

from typing import Dict,Any
from pandas.tseries.offsets import BDay
import pickle


DATA_DIR = config("DATA_DIR")


class FDAStrategy(Strategy):

    """ FDA-driven event strategy:
    - Entry on x day after FDA approval annoucnement
    - Exit after fixed holding period or on stop-loss
    - If at add_pos_days abs(add_pos_rate)> then, add up the position
    
    """
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        # params expects : 
        # {'approval_date':{"AAPL":pd.Timestamp("2020-05-01")...},'starting_day':1, holding_days':5,'stop_loss_pct':0.05,
        # 'trailing_loss_days':5, 'max_positions':4,'add_pos_days':5,'add_pos_rate':0.1}

        self.approvals = self.params['approval_date']
        self.starting_day = self.params['starting_day'] # number of days we start later than fda approval
        self.holding_days = self.params.get('holding_days', 20)
        self.stop_loss_pct = self.params.get('stop_loss_pct', 0.10) # stop loss for pct
        
        self.max_positions = self.params.get('max_positions', 4) # number of holdings at a time
        self.max_weight = 1 / (self.max_positions) # max weight per time
        
    def generate_signals(self, return_df: pd.DataFrame) -> pd.DataFrame:

        return_df = return_df.sort_index()
        dates    = return_df.index
        signals  = pd.DataFrame(0.0, index=dates, columns=return_df.columns)

        # build FIFO pending list
        pending = sorted(
        [(t, pd.to_datetime(dt)) for t, dts in self.approvals.items() for dt in dts],
        key=lambda x: x[1]
    )

        # 3) active 포지션 관리 구조
        #    ticker -> {'entry': entry_date, 'exit': exit_date}
        active = {}

        for current in dates:
            # A) Stop-loss 및 만기 청산
            for t, st in list(active.items()):
                entry = st['entry']
                exit_ = st['exit']

                # 1) 만기 지나면 청산
                if current > exit_:
                    active.pop(t)
                    continue

                # 2) stop-loss 체크: entry → current 누적 수익률
                cum_ret = (return_df[t]
                        .loc[entry:current]
                        .add(1)
                        .cumprod()
                        .sub(1))
                if cum_ret.iloc[-1] <= -self.stop_loss_pct:
                    active.pop(t)

            # B) 신규 진입: pending 중 현재 진입 가능 대상
            due = []
            for t, dt in pending:
                entry_date = dt + BDay(self.starting_day)
                if entry_date <= current:
                    due.append((t, dt, entry_date))
            # 날짜 순서대로, max_positions 제한 내에서 진입
            for t, dt, entry_date in due:
                if len(active) >= self.max_positions:
                    break
                # 이미 포지션 있으면 skip
                if t in active:
                    pending.remove((t, dt))
                    continue
                # 진입
                exit_date = entry_date + BDay(self.holding_days)
                # 백필(backfill)로 exit_date 맞추기
                idx = signals.index.get_indexer([exit_date], method='backfill')[0]
                exit_aligned = signals.index[idx]

                active[t] = {'entry': entry_date, 'exit': exit_aligned}
                pending.remove((t, dt))

            # C) signals 할당: active 중인 종목에 max_weight 할당
            for t, st in active.items():
                signals.at[current, t] = self.max_weight

        return signals



if __name__ == "__main__":
    ret_df = pd.read_parquet(DATA_DIR/"vht_dly_ret.parquet")
    # with open(DATA_DIR/"final_filing_dict.pkl",'rb') as f:
    #     temp = pickle.load(f)

    pp = Preprocessor(filing_dict_path = DATA_DIR / "full_classification1.pkl" , return_df_path= DATA_DIR/"vht_dly_ret.parquet")
    events = pp.sort_events(threshold=0.9)
    event_df= pd.DataFrame(events['New_Drug_Approval']).filter(items=['ticker','filedAt'])
    event_df['filedAt'] = pd.to_datetime(event_df['filedAt'], utc=True).dt.tz_convert(None).dt.normalize()
    approval_dict = (event_df.groupby('ticker')['filedAt'].apply(list).to_dict())

    params = {'approval_date':approval_dict,'starting_day':2, 'holding_days':60,'stop_loss_pct':0.1,
         'max_positions':2,'add_pos_days':5,'add_pos_rate':100}
    fda_strategy = FDAStrategy(params = params )  
    signals = fda_strategy.generate_signals(ret_df)
    strat = (signals * ret_df).sum(axis=1)
    import quantstats_lumi as qs
    qs.reports.html(strat,benchmark='^GSPC',output="fda.html")
    print(signals)

    ### 모든 파라미터에서 어떤 sharpe, dd 나오는지 check
    # run_backtest에서 하면 될듯..? 또 어떤 포지션에서 손실이 제일 많이 났고 이익이 많이 났는지 
  
    






    