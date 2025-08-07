from pull_fred import pull_fred
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from settings import config
from preprocessing import Preprocessor
import plotly.express as px
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
import pandas as pd
import numpy as np
from scipy import stats
import hypothesis_testing 
import plotly.subplots as sp
from sklearn.metrics import (classification_report, confusion_matrix, precision_score,recall_score, f1_score)
from typing import List

# visualize the the series
# Graph the series YOY, MOM etc.

# first original series, mom and then yoy 
DATA_DIR = config("DATA_DIR")
OUTPUT_DIR = config("OUTPUT_DIR")


def report_classification(path):
    # path jsonl path like JSON_CLASSIFICATION_DIR = DATA_DIR / "eval_jsonl_classifi.jsonl"
    item = pd.read_json(path,lines=True)
    df = pd.json_normalize(item['item'])
    y_pred = df['previous_pred'].to_list()
    y_true = df['correct_label'].to_list()
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    # Precision, Recall, F1
    precision = precision_score(y_true, y_pred, average='macro')
    recall    = recall_score(y_true, y_pred, average='macro')
    f1        = f1_score(y_true, y_pred, average='macro')

    print(f"\nMacro Precision: {precision:.2f}")
    print(f"Macro Recall:    {recall:.2f}")
    print(f"Macro F1:        {f1:.2f}")
    print(classification_report(y_true, y_pred))


def plot_raw_series_subplots(df, map_df, dimension=(10, 7)):
    id_list = df.columns.tolist()
    title_series = map_df.set_index('id').loc[id_list]['title']

    fig = make_subplots(
        rows=dimension[0], cols=dimension[1],
        subplot_titles=tuple(title_series.values)
    )

    for idx in range(len(df.columns)):
        i = idx // dimension[1] + 1  # subplot row index (1-based)
        j = idx % dimension[1] + 1   # subplot col index (1-based)

        series = df.iloc[:, idx]
        name = df.columns[idx]

        # connectgaps=True는 monthly/quarterly data gap 메우기
        if series.dropna().empty:
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    name=name,
                    mode="text",
                    text=["No Data"],
                    textposition="middle center"
                ),
                row=i, col=j
            )
        else:
            fig.add_trace(
                go.Scatter(x=df.index, y=series, name=name, connectgaps=True),
                row=i, col=j
            )

    fig.update_layout(height=2000, width=4800, title_text="All Series")

    return fig

def plot_acf_subplots(df, map_df, max_lag=12, dimension=(10, 7), alpha=0.05):
    id_list = df.columns.tolist()
    title_series = map_df.set_index('id').loc[id_list]['title']

    fig = make_subplots(
        rows=dimension[0],
        cols=dimension[1],
        subplot_titles=title_series.tolist()
    )

    for idx, col in enumerate(df.columns):
        series = df[col].dropna()
        valid_len = len(series)

        row = idx // dimension[1] + 1
        col_pos = idx % dimension[1] + 1

        if valid_len < 2:
            # Add dummy "No Data" trace to avoid subplot index errors
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="text",
                    text=["No Data"],
                    textposition="middle center",
                    showlegend=False
                ),
                row=row, col=col_pos
            )
            continue

        safe_lag = min(max_lag, valid_len - 1)
        acf_vals, confint = acf(series, nlags=safe_lag, alpha=alpha)

        lags = list(range(len(acf_vals)))
        lower = confint[:, 0]
        upper = confint[:, 1]

        # ACF bar plot
        fig.add_trace(
            go.Bar(x=lags, y=acf_vals, name="ACF", marker_color="steelblue"),
            row=row, col=col_pos
        )

        # CI band (transparent)
        fig.add_trace(
            go.Scatter(
                x=lags + lags[::-1],
                y=upper.tolist() + lower[::-1].tolist(),
                fill="toself",
                fillcolor="rgba(135, 206, 250, 0.3)",
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False,
                hoverinfo="skip"
            ),
            row=row, col=col_pos
        )

        # zero line
        fig.add_trace(
            go.Scatter(
                x=[0, safe_lag],
                y=[0, 0],
                mode="lines",
                line=dict(color="black", dash="dot"),
                showlegend=False
            ),
            row=row, col=col_pos
        )

    fig.update_layout(
        height=2000,
        width=4800,
        showlegend=False,
        title_text="Autocorrelation Plots with Confidence Intervals"
    )
    return fig

def plot_cumulative_event_returns(pp, events, dimension = (10,7)):
    """
    ** This takes in multiple events dict ** 
    This plots cumulative returns by events
    """

    combined_df = pd.DataFrame()

    for phase, recs in events.items(): # recs : list 
        if not recs:
            continue

        cum_df = pp.cumulative_event_window(recs,
                                            prev_window=60,
                                            post_window=60)
        
        avg_cum = cum_df.mean(axis=0)  # Series, average it by day

        # Make a DataFrame with phase as a column
        temp_df = pd.DataFrame({
            'days': avg_cum.index,
            'avg_cum_return': avg_cum.values,
            'event_type': phase
        })

        # Concatenate each df vertically
        combined_df = pd.concat([combined_df, temp_df], ignore_index=True)

    # Now use Plotly to plot all in one figure
    fig = px.line(combined_df,
                  x='days',
                  y='avg_cum_return',
                  color='event_type',
                  markers=True,
                  title='Event-anchored cumulative return by phase')

    fig.update_layout(
        xaxis_title='Business days relative to filing (0)',
        yaxis_title='Average cumulative return',
        template='plotly_white',
        legend_title='Event type'
    )

    fig.add_vline(x=0, line_dash="dash", line_color="gray")

    return fig

def plot_individual_cumulative_returns(pp, events, prev_window=0, post_window=20):
    """
    Plot cumulative returns for each firm per event type as individual lines, grouped by subplot.
    """
    cum_dict = {}
    for phase, recs in events.items():
        if not recs:
            continue
        cum_df = pp.cumulative_event_window({phase: recs},
                                            prev_window=prev_window,
                                            post_window=post_window)
        cum_dict[phase] = cum_df

    n_events = len(cum_dict)
    fig = make_subplots(rows=n_events, cols=1,
                        subplot_titles=list(cum_dict.keys()),
                        shared_xaxes=True)

    for i, (phase, df) in enumerate(cum_dict.items(), start=1):
        for firm in df.index:
            fig.add_trace(go.Scatter(x=df.columns,
                                     y=df.loc[firm],
                                     mode='lines',
                                     name=firm,
                                     showlegend=False),
                          row=i, col=1)

        # Add event-day line
        fig.add_vline(x=0, line_dash='dash', line_color='gray', row=i, col=1)

    fig.update_layout(height=1000 * n_events,
                      title='Cumulative Returns by Event Type and Firm',
                      template='plotly_white')

    fig.update_xaxes(title_text='Days from Event')
    fig.update_yaxes(title_text='Cumulative Return')
    return fig


def plot_event_frequency(events):
    """
    This plots number of events that is classified into specific events
    Returns histogram where x axis is a events and y is a number of filings sorted to such.
    """

    # Count how many records for each event type
    event_counts = {event_type: len(recs) for event_type, recs in events.items() if event_type !="MISC"}

    # Convert to DataFrame
    freq_df = pd.DataFrame({
        "event_type": list(event_counts.keys()),
        "frequency": list(event_counts.values())
    })

    # Plot using plotly express
    fig = px.bar(freq_df,
                 x="event_type",
                 y="frequency",
                 title="Number of filings per event type",
                 text="frequency")

    fig.update_layout(
        xaxis_title="Event Type",
        yaxis_title="Number of Filings",
        template="plotly_white"
    )

    return fig

def plot_volatility(raw,est_t1 = -250, est_t2 = -31):
    # This plots annualized volatility over est_t1 ~ est_t2 for each events
    # Purpose of this plot is to better interpret SCAR statistics and it's decision

    event_names = list(raw.keys()) #'PHASE1_neg...,'
    rows = len(event_names)

    fig = make_subplots(
        rows=rows,
        cols=1,
        subplot_titles=[f"{event}" for event in event_names],
        vertical_spacing=0.05
    )

    for i, (key, val) in enumerate(raw.items(), start=1):
        est_window = val.loc[:, est_t1:est_t2]
        vol_annualized = est_window.std(axis=1, ddof=1).values * np.sqrt(252)

        fig.add_trace(
            go.Scatter(
                x=list(range(len(vol_annualized))),
                y=vol_annualized,
                mode="markers",
                text=val.index,
                name=key,
                hovertemplate="Ticker: %{text}<br>Vol: %{y:.2%}<extra></extra>"
            ),
            row=i,
            col=1
        )

    fig.update_layout(
        height=1000*rows,  
        showlegend=False,
        title_text="Annualized Volatility by Event Type",
        template="plotly_white"
    )

    fig.update_xaxes(title_text="Ticker (index order)")
    fig.update_yaxes(title_text="Annualized Volatility")

    return fig

def plot_pvalue_evolution(raw, max_window=60, lag = 2, scaled=False):

    pos_matrix, neg_matrix = hypothesis_testing.rolling_pvalues(raw, max_window=max_window, lag = lag, scaled=scaled)
    
    fig = go.Figure()

    # Positive P-values
    for evt in pos_matrix.index:
        fig.add_trace(go.Scatter(
            x=pos_matrix.columns,
            y=pos_matrix.loc[evt],
            mode='lines+markers',
            name=f'{evt} (pos)',
            line=dict(dash='solid')
        ))

    # Negative P-values
    for evt in neg_matrix.index:
        fig.add_trace(go.Scatter(
            x=neg_matrix.columns,
            y=neg_matrix.loc[evt],
            mode='lines+markers',
            name=f'{evt} (neg)',
            line=dict(dash='dash')
        ))

    # Significance level line
    fig.add_shape(
        type='line',
        x0=1,
        x1=max_window,
        y0=0.05,
        y1=0.05,
        line=dict(color='red', dash='dot'),
    )

    fig.update_layout(
        title="One-Sided P-value Evolution Over Event Window",
        xaxis_title="Days in Event Window (Post-Release)",
        yaxis_title="P-value",
        yaxis=dict(range=[0, 0.3]),
        template="plotly_white",
        legend_title="Event Type (Pos/Neg)"
    )

    return fig

def plot_walk_forward_pvalue_evolution(agg_dict,windows:List):
    
    fig = go.Figure()
    for event, df in agg_dict.items():
        for t in windows:
            if t in df.columns:
                y = df[t].astype(float)
                fig.add_trace(go.Scatter(
                    x=df.index, y=y,
                    mode='lines+markers',
                    name=f"{event} (window={t})",
                    legendgroup=event,
                    showlegend=True
                ))
    fig.add_shape(
        type="line",
        x0=min(df.index), x1=max(df.index), y0=0.05, y1=0.05,
        line=dict(color="red", width=2, dash="dash"),
        xref="x", yref="y"
    )
    fig.update_layout(
        title="Walk forward Rolling P-Value",
        xaxis_title="Year",
        yaxis_title="P-Value",
        legend_title="Event (window)",
        template='plotly_white',
        hovermode="x unified"
    )   
    
    return fig

if __name__=="__main__":
    filings = DATA_DIR / "filings_dict.pkl"
    dly_ret = DATA_DIR / "vht_dly_ret.parquet"




   


