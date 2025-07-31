import pandas as pd
import numpy as np
from linearmodels.panel import FamaMacBeth
from linearmodels import OLS
from scipy import stats
import statsmodels.api as sm

class FamaMacBethRollingLM:
    def __init__(
        self,
        dependent: pd.DataFrame,  # y: (dates × tickers)
        factors: pd.DataFrame,    # factor returns: (dates × factors)
        window: int,
    ):
        self.y = dependent
        self.factors = factors
        self.window = window

    def fit(self, cov_type="robust"):
        dates = self.y.index
        tickers = self.y.columns
        factor_names = self.factors.columns

        betas_rolling = {}
        lambda_list = []
        lambda_dates = []

        # Time-Series regression to calculate rolling betas
        for i in range(self.window, len(dates)):
            end_date = dates[i]
            start_date = dates[i - self.window]
            y_window = self.y.loc[start_date:end_date]
            factors_window = self.factors.loc[start_date:end_date]

            betas_period = pd.DataFrame(index=tickers, columns=factor_names, dtype=float)

            for ticker in tickers:
                y_stock = y_window[ticker].dropna()
                common_dates = y_stock.index.intersection(factors_window.index)
                y_stock = y_stock.loc[common_dates]
                x_fct = factors_window.loc[common_dates]

                if len(y_stock) < self.window:
                    continue

                x_fct_const = sm.add_constant(x_fct)
                model = OLS(y_stock, x_fct_const).fit()
                betas_period.loc[ticker] = model.params[1:]  # drop constant

            betas_rolling[end_date] = betas_period

        # Cross-sectional regression (Fama-MacBeth)
        for date in dates[self.window:]:
            betas_period = betas_rolling[date]
            y_cross = self.y.loc[date].dropna()
            valid_tickers = y_cross.index.intersection(betas_period.dropna().index)

            K = betas_period.shape[1]
            
            if len(valid_tickers) == 0:
                continue
            
            if len(valid_tickers) <= K:
                continue

            y_cross = y_cross.loc[valid_tickers]
            beta_cross = betas_period.loc[valid_tickers]

            y_panel = y_cross.rename_axis('ticker').to_frame(name='excess_return')
            y_panel['date'] = date
            y_panel = y_panel.reset_index().set_index(['ticker', 'date'])

            beta_panel = beta_cross.copy()
            beta_panel['date'] = date
            beta_panel = beta_panel.reset_index().set_index(['ticker', 'date'])

            # Call FamaMacBeth
            fm = FamaMacBeth(y_panel['excess_return'], beta_panel)
            res = fm.fit(cov_type=cov_type)

            lambda_list.append(res.params)
            lambda_dates.append(date)

        # lambda time series results
        lambdas = pd.DataFrame(lambda_list, index=lambda_dates)

        # lambda stats
        lambda_mean = lambdas.mean()
        lambda_cov = lambdas.cov()
        lambda_se = np.sqrt(np.diag(lambda_cov / len(lambdas)))
        lambda_tstats = lambda_mean / lambda_se
        lambda_pvals = 2 * (1 - stats.t.cdf(np.abs(lambda_tstats), df=len(lambdas)-1))

        results = {
            "rolling_betas": betas_rolling,
            "factor_premiums": lambda_mean,
            "std_err": lambda_se,
            "t_stats": lambda_tstats,
            "p_values": lambda_pvals,
            "cov": lambda_cov / len(lambdas),
            "factor_premiums_timeseries": lambdas,
        }

        return results

def shanken_correction_LM(results: dict, threshold_t_over_n=3, tstat_threshold=3.0):
    lambdas = results["factor_premiums_timeseries"]
    
    # Concatenate all betas vertically
    betas_all = pd.concat(results["rolling_betas"].values(), axis=0).dropna()

    T = lambdas.shape[0]  # periods
    N = betas_all.shape[0]  # num of betas (ticker-periods)
    
    lambda_mean = lambdas.mean().values.reshape(-1, 1)
    Sigma_lambda = lambdas.cov().values
    Sigma_beta = betas_all.cov().values

    std_err = results["std_err"]
    t_stats = results["t_stats"]

    apply_shanken = (T/N < threshold_t_over_n) or (np.any(np.abs(t_stats) > tstat_threshold))

    if apply_shanken:
        adj = (1 + lambda_mean.T @ np.linalg.inv(Sigma_beta) @ lambda_mean).item()
        corrected_cov = Sigma_lambda/T + (adj/N)*Sigma_beta
        corrected_std_err = np.sqrt(np.diag(corrected_cov))
        corrected_tstats = lambda_mean.flatten() / corrected_std_err
        corrected_pvals = 2 * (1 - stats.t.cdf(np.abs(corrected_tstats), df=T-1))

        correction_applied = True
    else:
        corrected_cov = results["cov"]
        corrected_std_err = std_err
        corrected_tstats = t_stats
        corrected_pvals = results["p_values"]
        correction_applied = False

    corrected_results = {
        "apply_shanken": correction_applied,
        "corrected_cov": corrected_cov,
        "corrected_std_err": corrected_std_err,
        "corrected_t_stats": corrected_tstats,
        "corrected_p_values": corrected_pvals,
        "original_std_err": std_err,
        "original_t_stats": t_stats,
        "original_p_values": results["p_values"]
    }

    return corrected_results