from pull_fred import pull_fred
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from settings import config
from preprocessing import Preprocessor
import plotly.express as px
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
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



if __name__=="__main__":
    print("HI")
    macro_map_dir = DATA_DIR/"macro_map.parquet"
    macro_latest_series = DATA_DIR/"macro_latest_series.parquet"

    macro_map = Preprocessor(macro_map_dir).get()
    macro_latest_series = Preprocessor(macro_latest_series).get()
    fig = plot_raw_series_subplots(macro_latest_series, macro_map)
    fig.write_html(OUTPUT_DIR/"macro_raw.html")

    fig2 = plot_acf_subplots(macro_latest_series, macro_map)
    fig2.write_html(OUTPUT_DIR/"macro_acf.html")


