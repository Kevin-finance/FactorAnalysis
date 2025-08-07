# run_backtest에서 하면 될듯..? 또 어떤 포지션에서 손실이 제일 많이 났고 이익이 많이 났는지 
    ### 모든 파라미터에서 어떤 sharpe, dd 나오는지 check
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import quantstats_lumi as qs
import pandas as pd
import plotly.graph_objs as go


def plot_return_distribution(strat):

    sns.histplot(strat, bins=40, kde=True)
    mu, sigma = strat.mean(), strat.std()

    sns.histplot(strat[np.abs(strat)<0.01], bins=200, kde=True, stat="density", color='skyblue')
    plt.axvline(0, color='red', linestyle='--')
    plt.title("Position return distribution")

    # Normal PDF
    x = np.linspace(strat.min(), strat.max(), 200)
    plt.plot(x, norm.pdf(x, mu, sigma), 'k--', label="Normal PDF")
    plt.legend()
    plt.ylim((0,2000))
    plt.xlim(-0.1,0.1)
    plt.axvline(0, color='red', linestyle='--') 
    plt.title("Position return distribution")
    plt.show()

def performance_metrics(signal,ret_df,benchmark):
    # Both in series
    perf_df = pd.DataFrame(columns = ['Correlation','Sharpe','Active_Sharpe','CAGR',"MDD",'Skew','Kurtosis'])
    
    strat = (signal * ret_df).sum(axis=1)
    corr_with_bm = qs.stats.benchmark_correlation(strat.to_frame(),benchmark=benchmark).values[0]
    sharpe = qs.stats.sharpe(strat)
    cagr = qs.stats.cagr(strat)
    mdd = qs.stats.drawdown_details(strat).sort_values(by='max drawdown')['max drawdown'].iloc[0]
    skew = qs.stats.skew(strat)
    kurtosis = qs.stats.kurtosis(strat)
    print(corr_with_bm,sharpe,cagr,mdd,skew,kurtosis)
    mask_active = signal.sum(axis=1) > 0
    active_ret = strat[mask_active]
    in_exposure_sharpe = qs.stats.sharpe(active_ret)

    perf_dict = {
    'Correlation': corr_with_bm,
    'Sharpe': sharpe,
    'Active_Sharpe': in_exposure_sharpe,
    'CAGR': cagr,
    'MDD': mdd,
    'Skew': skew,
    'Kurtosis': kurtosis,
}

    perf_df = pd.DataFrame([perf_dict])
    return perf_df
    

def number_of_holdings(signal,max_position):
    n_positions = signal.sum(axis=1)*max_position

    return n_positions

    # cum_ret = (strat + 1).cumprod() - 1
    # n_positions = signal.sum(axis=1)*3

    # fig = go.Figure()

    # fig = make_subplots(specs=[[{"secondary_y": True}]])

    # # 누적수익률 (왼쪽 y축, 파란 꺾은선)
    # fig.add_trace(
    #     go.Scatter(
    #         x=cum_ret.index, y=cum_ret.values,
    #         name='Cumulative Return',
    #         line=dict(width=3, color='blue')
    #     ),
    #     secondary_y=False,
    # )

    # # 포지션 수 (오른쪽 y축, 진한 파랑 bar)
    # fig.add_trace(
    #     go.Bar(
    #         x=n_positions.index, y=n_positions.values,
    #         name='Number of Positions',
    #         marker=dict(color='rgba(30,50,0,60)'),
    #         opacity=0.3
    #     ),
    #     secondary_y=True,
    # )

    # fig.update_layout(
    #     title='누적수익률 & 동시 포지션 수 (오른쪽 y축, Plotly)',
    #     height=500,
    #     bargap=0,
    #     legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    # )

    # fig.update_xaxes(title_text='Date')
    # fig.update_yaxes(title_text='누적수익률', secondary_y=False)
    # fig.update_yaxes(title_text='Number of Positions', secondary_y=True)

    # fig.show()
def top_winners_losers(ret_df,top_p,bottom_q):
    pos_df = ret_df.replace(0, pd.NA)  
    stacked = pos_df.stack()      
    returns = stacked.dropna().astype(float)

    top_q = returns.quantile(top_p)
    bot_q = returns.quantile(bottom_q)

    top10 = returns[returns >= top_q]
    bot10 = returns[returns <= bot_q]

    top10_df = top10.reset_index()
    top10_df.columns = ['Exposure_date', 'ticker', 'return']

    bot10_df = bot10.reset_index()
    bot10_df.columns = ['Exposure_date', 'ticker', 'return']


    return top10_df.sort_values(by='return',ascending=False) , bot10_df.sort_values(by='return',ascending=True)
