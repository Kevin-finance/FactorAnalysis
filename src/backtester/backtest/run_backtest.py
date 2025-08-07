import pandas as pd
from backtester.strategy.event_fda_strategy import FDAStrategy
import backtester.backtest.metrics as metric
from backtester.config import parameter_grid
from preprocessing import Preprocessor
from settings import config 
import quantstats_lumi as qs
import os
import yfinance as yf
os.environ['PYTHONUTF8'] = '1'

DATA_DIR = config("DATA_DIR")
OUTPUT_DIR = config("OUTPUT_DIR")


# cmd : python -X utf8 -m backtester.backtest.run_backtest
benchmark = yf.download("^GSPC",start = "2014-12-30",end="2024-12-31")
bm_ret = benchmark.pct_change().dropna()

def run(return_df,param_set,benchmark):
    # benchmark is a series
    # each param_set is a list of dict
    strategy = FDAStrategy(params = param_set)
    signal = strategy.generate_signals(return_df) # weights
    strat = (signal * return_df).sum(axis=1)

    return signal, strat



