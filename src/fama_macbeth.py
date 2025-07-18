import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from typing import Optional

class FamaMacBethRolling:
    def __init__(
        self,
        dependent: pd.DataFrame,  # y: index date, cols ticker, stk excess rtn (r-rf)
        exog: pd.DataFrame,       # x: index date, cols factors, fct rtn
        window: Optional[int] = None, # default to be None
    ):
        self.y = dependent
        self.x = exog
        self.window = window

    def fit(self, cov_type="robust"):
        y, x = self.y, self.x
        dates = y.index
        tickers = y.columns

        betas_rolling = {}
        lambdas = []

        # Time Series regression, cal rolling beta
        for i in range(self.window, len(dates)):
            end_date = dates[i]
            
            # check if window is set
            if self.window is None:
                start_date = dates[0]
                y_window = y.loc[start_date:end_date] # directly use the past data
                x_window = x.loc[start_date:end_date]
            else:
                if i < self.window:
                    continue
                start_date = dates[i - self.window]
                y_window = y.loc[start_date:end_date]
                x_window = x.loc[start_date:end_date]

            betas_period = pd.DataFrame(index=tickers, columns=x.columns, dtype=float)

            for ticker in tickers:
                y_stock = y_window[ticker].dropna()
                common_dates = y_stock.index.intersection(x_window.index)
                y_stock = y_stock.loc[common_dates]
                x_fct = x_window.loc[common_dates]

                if len(y_stock) < self.window:
                    continue

                x_fct_const = sm.add_constant(x_fct)
                model = sm.OLS(y_stock, x_fct_const).fit()
                betas_period.loc[ticker] = model.params[1:]  # drop constant

            betas_rolling[end_date] = betas_period

        # Cross-sectional regression
        lambda_list = []
        lambda_dates = []

        for date in dates[self.window:]:
            betas_period = betas_rolling[date]
            y_cross = y.loc[date].dropna()
            valid_tickers = y_cross.index.intersection(betas_period.dropna().index)

            if len(valid_tickers) == 0:
                continue

            y_cross = y_cross.loc[valid_tickers]
            beta_cross = betas_period.loc[valid_tickers]

            beta_cross_const = sm.add_constant(beta_cross)
            model = sm.OLS(y_cross, beta_cross_const).fit()

            lambda_list.append(model.params[1:])  # drop constant
            lambda_dates.append(date)

        lambdas = pd.DataFrame(lambda_list, index=lambda_dates)

        # lambda stats, mean, cov, stderror, tstat, pvalue
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
            "rolling_betas": betas_rolling,
        }

        return results


def shanken_correction(results: dict, threshold_t_over_n=3, tstat_threshold=3.0):
    lambdas = results["factor_premiums_timeseries"]
    
    # Concatenate betas across all periods vertically
    betas_all = pd.concat(results["rolling_betas"].values(), axis=0).dropna()

    T = lambdas.shape[0]  # num of timeperiods
    N = betas_all.shape[0]  # num of β = tickeramount * T
    
    lambda_mean = lambdas.mean().values.reshape(-1,1)  # (K × 1)
    Sigma_lambda = lambdas.cov().values                # (K × K)
    Sigma_beta = betas_all.cov().values                # (K × K)

    std_err = results["std_err"]
    t_stats = results["t_stats"]

    # condition to apply shanken correction
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