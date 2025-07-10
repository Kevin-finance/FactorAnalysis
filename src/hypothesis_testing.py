import numpy as np
from scipy import stats
from settings import config
from preprocessing import Preprocessor

DATA_DIR = config("DATA_DIR")

def t_test(raw, window_t1, window_t2):
    hypothesis_dict = {}
    window_length = window_t2 - window_t1 + 1
    for key, df in raw.items():
        event_window = df.loc[:, window_t1:window_t2]
        CARs = event_window.sum(axis=1)

        CAR_mean = CARs.mean()

        # This assumes that mean of returns are independent
        # Check for SE
        SE = np.std(event_window.values.sum(axis=1), ddof=1) / np.sqrt(len(event_window))

        t_stat = CAR_mean / (SE + 1e-8)
        dfree = len(CARs) - 1 

        # Two-sided p-value
        two_sided_pval = stats.t.sf(np.abs(t_stat), df=dfree) * 2

        # One-sided p-values (fixed definitions)
        one_sided_positive_pval = stats.t.sf(t_stat, df=dfree)   # H₁: CAR > 0
        one_sided_negative_pval = stats.t.cdf(t_stat, df=dfree)  # H₁: CAR < 0

        hypothesis_dict[key] = {
            'CAR': CAR_mean,
            'SE': SE,
            't-stat': t_stat,
            'two_sided_pval': two_sided_pval,
            'one_sided_positive_pval': one_sided_positive_pval,
            'one_sided_negative_pval': one_sided_negative_pval,
        }

    return hypothesis_dict












if __name__ == "__main__": 
    from preprocessing import Preprocessor
    filings = DATA_DIR / "filings_dict.pkl"
    dly_ret = DATA_DIR / "vht_dly_ret.parquet"

    pp      = Preprocessor(DATA_DIR/"filings_dict.pkl", DATA_DIR/"vht_dly_ret.parquet")
    events = pp.sort_events()
    raw = pp.raw_event_window(events, prev_window=0, post_window=5)
    temp = t_test(raw,window_t1 = 0 , window_t2=3)
    print(temp)
    
 