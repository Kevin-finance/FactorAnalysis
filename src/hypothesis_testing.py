import numpy as np
from scipy import stats
from settings import config
from preprocessing import Preprocessor
from itertools import combinations
from scipy.stats import norm
import pandas as pd

DATA_DIR = config("DATA_DIR")

def t_test(raw, window_t2 = 1, window_t1=0, est_t1 = -250, est_t2 = -31, cross_dependence=True, scaled=True):
    """
    Not accounting for cross-sectional dependence, if its accepted then it is true accept
    However, if its rejected there is possibility of acceptance with robust s.e(Type1 error)
    
    """
    print(raw)

    hypothesis_dict = {} 
    for key, df in raw.items(): # {'MISC':df...}
        
        # 1) Estimation window for SCAR
        if scaled: 
            est_window = pd.DataFrame(df).loc[:,est_t1:est_t2] # using estimation window -250 ~ -31 (prevent info leakage)
            std = np.nanstd(est_window,axis=1,ddof=1) # ignore nan, there are stocks that are not yet listed or no data for event window we set
            # std_est_window= ((est_window.T)/std).T # standardized returns

        # 2) Event period for CAR and SCAR 
        event_window = pd.DataFrame(df).loc[:, window_t1:window_t2]
        CARs = event_window.sum(axis=1) # summing over time (cumulative in time axis and for same firm)
        SCARs = CARs / std if scaled else None # scaling CAR by its estimation window
        
        # 3) Taking mean and this statistics will go to the numerator
        CAR_mean = CARs.mean() 
        SCAR_mean = SCARs.mean() if scaled else None


        N = event_window.shape[0] # Number of firms

        # 4) Cross-dependence adjustment
        # if not and has positive cross-dependence then, sample std is negatively biased -> t-inflated
        if cross_dependence:

            # Correlation with each rows(firms)
            corr_matrix = event_window.T.corr().values # N*N matrix

            triu_idx = np.triu_indices_from(corr_matrix, k=1) # index of upper triangular matrix (tuple)
                  
            rho_bar = np.nanmean(corr_matrix[triu_idx]) # mean of strictly upper trinagular elements(no diagonal)
            # Note : If event window = 1, then corr = 1 and if event # <=1 then warning
        else:
            rho_bar = 0.0 

        # 5) Standard errors
        if scaled:
            # Kolari & Pynnönen ADJ-BMP SE for SCARs
            s2 = SCARs.var(ddof=1)
            theta = rho_bar  # under null, theta ≈ rho_bar

            if theta > 0:
                SE = np.sqrt((s2 / (1 - theta)) * (1 + (N - 1) * theta) / N)
            else:
                SE = np.sqrt(s2 / N)
            t_stat = SCAR_mean / SE
        else:
            # Naive or Kolari CAR SE
            se_unadjusted = CARs.std(ddof=1) / np.sqrt(N) 
            if rho_bar > 0:
                SE = se_unadjusted * np.sqrt(1 + (N - 1) * rho_bar)
            else: 
                SE = se_unadjusted

            t_stat = CAR_mean / SE
    
        # Two-sided p-value
        two_sided_pval = stats.t.sf(np.abs(t_stat), df=N-1) * 2 # H0: CAR_bar = 0 

        # One-sided p-values 
        one_sided_positive_pval = stats.t.sf(t_stat, df=N-1)   # H₁: CAR_bar > 0 / Tests for positive drift of PHASE1/2/3, FDA_APPROVAL
        one_sided_negative_pval = stats.t.cdf(t_stat, df=N-1)  # H₁: CAR_bar < 0 / Tests for CRL mainly


        hypothesis_dict[key] = {
            'CAR': CAR_mean,
            'SCAR':SCAR_mean if scaled else None,
            'SE': SE,
            't-stat': t_stat,
            'two_sided_pval': two_sided_pval,
            'one_sided_positive_pval': one_sided_positive_pval,
            'one_sided_negative_pval': one_sided_negative_pval,
        }

    return hypothesis_dict

def rolling_pvalues(raw,max_window=60, lag = 2, scaled=False):
    # filedAt got rid of time thus lag 2 is a starting point

    pos_matrix = pd.DataFrame()
    neg_matrix = pd.DataFrame()

    for t in range(lag+1, max_window + 1):
        print(f'rolling p lag:{t} done')
        # Takes in dict {'MISC',df...} where df has index of tickers_1,,, col: relative date to filing , values : returns
        result = t_test(raw, window_t1= lag, window_t2=t, scaled=scaled) 
        
        pos_values = {evt: result[evt].get("one_sided_positive_pval", np.nan) for evt in result}
        neg_values = {evt: result[evt].get("one_sided_negative_pval", np.nan) for evt in result}
        
        pos_matrix[t] = pd.Series(pos_values) 
        neg_matrix[t] = pd.Series(neg_values)

    return pos_matrix,neg_matrix







if __name__ == "__main__": 
    filings = DATA_DIR / "filings_dict.pkl"
    dly_ret = DATA_DIR / "vht_dly_ret.parquet"

    pp      = Preprocessor(DATA_DIR/"filings_dict.pkl", DATA_DIR/"vht_dly_ret.parquet")
    events = pp.sort_events()
    raw = pp.raw_event_window(events, prev_window=250, post_window=20)
    # temp = t_test(raw, window_t1 = 0 , window_t2=20)
    result_scaled = t_test(raw, window_t1=0, window_t2=10, scaled=True)
    result_unscaled = t_test(raw, window_t1=0, window_t2=10, scaled=False)

    for event in result_scaled:
        print(f"{event}:")
        print(f"  CAR (scaled): {result_scaled[event]['CAR']:.8f}")
        print(f"  CAR (unscaled): {result_unscaled[event]['CAR']:.8f}")
        print(f"  SCAR (scaled): {result_scaled[event]['SCAR']:.8f}")
        print(f"  SCAR (unscaled): {result_unscaled[event]['SCAR']}")  # Should be None
        print(f"  t-stat (scaled): {result_scaled[event]['t-stat']:.8f}")
        print(f"  t-stat (unscaled): {result_unscaled[event]['t-stat']:.8f}")
        print()
    
 