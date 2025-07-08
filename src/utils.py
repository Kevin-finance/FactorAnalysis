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
from hypothesis_testing import t_test
# visualize the the series
# Graph the series YOY, MOM etc.

# first original series, mom and then yoy 
DATA_DIR = config("DATA_DIR")
OUTPUT_DIR = config("OUTPUT_DIR")

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


# def plot_acf(df,map_df,dimension=(10,5)): # check acf, and stationarity by ADF
#     # pmdarima get rid of nan values internally

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
    This plots cumulative returns by events
    """

    combined_df = pd.DataFrame()

    for phase, recs in events.items():
        if not recs:
            continue
        cum_df = pp.cumulative_event_window({phase: recs},
                                            prev_window=60,
                                            post_window=60)
        avg_cum = cum_df.mean(axis=0)  # Series

        # Make a DataFrame with phase as a column
        temp_df = pd.DataFrame({
            'days': avg_cum.index,
            'avg_cum_return': avg_cum.values,
            'event_type': phase
        })
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




def plot_event_frequency(pp,events):
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

def plot_pvalue_evolution(raw, max_window=20):
    pos_matrix = pd.DataFrame()
    neg_matrix = pd.DataFrame()

    for t in range(1, max_window + 1):
        result = t_test(raw, window_t1=0, window_t2=t)
        
        # 안전하게 키 접근
        pos_values = {evt: result[evt].get("one_sided_positive_pval", np.nan) for evt in result}
        neg_values = {evt: result[evt].get("one_sided_negative_pval", np.nan) for evt in result}
        
        pos_matrix[t] = pd.Series(pos_values)
        neg_matrix[t] = pd.Series(neg_values)

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
        y0=0.95,
        y1=0.95,
        line=dict(color='red', dash='dot'),
    )

    fig.update_layout(
        title="One-Sided P-value Evolution Over Event Window",
        xaxis_title="Days in Event Window (Post-Release)",
        yaxis_title="P-value",
        yaxis=dict(range=[0.85, 1]),
        template="plotly_white",
        legend_title="Event Type (Pos/Neg)"
    )

    return fig

def main():
    # Macro related
    macro_map_dir = DATA_DIR/"macro_map.parquet"
    macro_latest_series_dir = DATA_DIR/"macro_latest_series.parquet"

    macro_map = Preprocessor(macro_map_dir).get()[macro_map_dir.name]
    macro_latest_series = Preprocessor(macro_latest_series_dir).get()[macro_latest_series_dir.name]
    fig = plot_raw_series_subplots(macro_latest_series, macro_map)
    fig.write_html(OUTPUT_DIR/"macro_raw.html")

    fig2 = plot_acf_subplots(macro_latest_series, macro_map)
    fig2.write_html(OUTPUT_DIR/"macro_acf.html")
    # Macro related end

    

if __name__=="__main__":
    filings = DATA_DIR / "filings_dict.pkl"
    dly_ret = DATA_DIR / "vht_dly_ret.parquet"

    pp      = Preprocessor(DATA_DIR/"filings_dict.pkl", DATA_DIR/"vht_dly_ret.parquet")
    events = pp.sort_events()
    raw = pp.raw_event_window(events, prev_window=0, post_window=20)
    plot_cumulative_event_returns(pp,events).write_html(OUTPUT_DIR/"cumulative_event_returns.html")
    plot_event_frequency(pp,events).write_html(OUTPUT_DIR/"event_frequencies.html")
    plot_pvalue_evolution(raw,max_window=20).write_html(OUTPUT_DIR/"evolving_pvalues.html")


   


