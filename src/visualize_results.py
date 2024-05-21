import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import bar_chart_race as bcr
import plotly.express as px
import warnings

pd.options.mode.chained_assignment = None
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)


# Plots for exploratory data analysis
def save_histogram_plot(
    data, title, x_label, y_label, save_path, log_scale=True, fig_size=(10, 6)
):
    plt.figure(figsize=fig_size)
    plt.hist(np.array(data), log=log_scale)
    plt.title(title)
    plt.xlabel(x_label, fontsize=13)
    plt.ylabel(y_label, fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.xticks(rotation=45)
    plt.savefig(save_path)
    plt.clf()


def plot_vars_distribution():
    print("Generating graphs for variables distribution...")
    # Reading the csv file
    file_path = "data/processed/time_series_data.csv"
    df = pd.read_csv(file_path, index_col=0)
    df["time"] = df.index

    data = df.magnitude1
    title = "Magnitude Vs Frequency"
    x_label = "EQ magnitude"
    y_label = "Frequency"
    save_path = "results/exploratory_data_analysis/EQ_magnitude_graph.png"

    data1 = df.depth
    title1 = "Depth Vs Frequency"
    x_label1 = "EQ depth"
    y_label1 = "Frequency"
    save_path1 = "results/exploratory_data_analysis/EQ_depth_graph.png"

    data2 = df.longitude
    title2 = "Longitude Vs Frequency"
    x_label2 = "EQ longitude"
    y_label2 = "Frequency"
    save_path2 = "results/exploratory_data_analysis/EQ_longitude_graph.png"

    data3 = df.latitude
    title3 = "Latitude Vs Frequency"
    x_label3 = "EQ latitude"
    y_label3 = "Frequency"
    save_path3 = "results/exploratory_data_analysis/EQ_latitude_graph.png"

    save_histogram_plot(data, title, x_label, y_label, save_path)
    save_histogram_plot(data1, title1, x_label1, y_label1, save_path1)
    save_histogram_plot(data2, title2, x_label2, y_label2, save_path2)
    save_histogram_plot(data3, title3, x_label3, y_label3, save_path3)
    print("Finished!")


def data_for_various_plots():
    print("Plotting various exploratory data analysis...")
    # Reading the csv file
    file_path = "data/processed/time_series_data.csv"
    df = pd.read_csv(file_path, index_col=0)
    df["time"] = df.index
    df_eq = df[["time", "magnitude1", "longitude", "latitude", "depth"]]
    df_eq["time"] = pd.to_datetime(df_eq["time"])
    df_eq["timestamps"] = df_eq["time"]

    # Feature Engineering
    # Number 1: Time intervals between consecutive earthquakes.
    df_eq["time_diff"] = df_eq["timestamps"].diff()
    df_eq["magnitude1"] = pd.to_numeric(df_eq["magnitude1"], errors="coerce")
    df_eq["time_diff_float"] = df_eq["time_diff"].apply(lambda x: x.total_seconds())
    # Number 2: the rolling of magnitudes from the last 10 earthquakes
    df_eq["mag_roll_10"] = df_eq["magnitude1"].rolling(window=10).mean()
    df_eq.dropna(inplace=True)
    # Labeling the large earthquakes (M>6). This helps slice the time-series for later specific purposes
    label = []
    cnt = 0
    for i, mag in enumerate(df_eq["magnitude1"]):
        if mag > 6:
            cnt = cnt + 1
            label.append(int(cnt))
        else:
            label.append(0)

    df_eq["large_eq_label"] = label
    # Checking the entirety of the time-series
    plt.title("Time Series Plot")
    df_eq.plot(subplots=True, figsize=(14, 8))
    plt.savefig("results/exploratory_data_analysis/EQ_time_series_graph.png")
    plt.clf()
    # collinearity between parameters
    df_corr = df_eq.corr()
    plt.figure(figsize=(8, 6))
    # heatmap
    plt.title("Heatmap")
    sns.heatmap(df_corr, vmin=-1, cmap="coolwarm", annot=True, fmt=".2f", square=True)
    plt.subplots_adjust(left=0.01, bottom=0.2, right=0.95, top=0.95,)
    plt.savefig("results/exploratory_data_analysis/EQ_heatmap_graph.png")
    plt.clf()
    plt.figure(figsize=(8, 6))
    # hist between longitude & latitude
    df_eq_plot = df_eq[df_eq["depth"] < 50]
    plt.hist2d(df_eq_plot["longitude"], df_eq_plot["latitude"], bins=(50, 50), vmax=400)
    plt.colorbar()
    plt.title("Longitude Vs Latitude")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.savefig("results/exploratory_data_analysis/EQ_hist1_graph.png")
    plt.clf()
    # Hist between magnitude & depth
    df_eq_plot = df_eq[df_eq["depth"] < 50]
    plt.hist2d(df_eq_plot["magnitude1"], df_eq_plot["depth"], bins=(50, 50), vmax=400)
    plt.colorbar()
    plt.title("Magnitude Vs Depth")
    plt.xlabel("EQ magnitude")
    plt.ylabel("Depth (km)")
    plt.savefig("results/exploratory_data_analysis/EQ_hist2_graph.png")
    plt.clf()
    print("Finished!")


def visualize_analysis_by_district():
    print("Starting to visualize analysis by district...")
    df = pd.read_csv(
        "data/analyzed/monthly_magnitude_by_district.csv", parse_dates=["date"]
    ).set_index("date")
    save_path = "results/analysis_by_district/magnitude.gif"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    bcr.bar_chart_race(
        df=df,
        filename=save_path,
        orientation="h",
        sort="desc",
        n_bars=8,
        fixed_order=False,
        fixed_max=False,
        steps_per_period=3,
        interpolate_period=False,
        label_bars=True,
        bar_size=0.95,
        period_label={"x": 0.99, "y": 0.25, "ha": "right", "va": "center"},
        period_fmt="%B, %Y",
        period_summary_func=lambda v, r: {
            "x": 0.99,
            "y": 0.18,
            "s": f"Total magnitude: {v.nlargest(8).sum():,.0f}",
            "ha": "right",
            "size": 8,
            "family": "Courier New",
        },
        perpendicular_bar_func="median",
        period_length=200,
        figsize=(5, 3),
        dpi=144,
        cmap="dark12",
        title="Magnitude by District",
        title_size="",
        bar_label_size=7,
        tick_label_size=7,
        shared_fontdict=None,
        scale="linear",
        writer=None,
        fig=None,
        bar_kwargs={"alpha": 0.7},
        filter_column_colors=False,
    )
    print("Finished!")


def plot_large_earthquakes():
    print("Plotting large earthquakes data...")
    df = pd.read_csv(
        "data/analyzed/time_series_data_lt6.csv", parse_dates=["datetime"]
    ).set_index("datetime")
    plt.figure(figsize=(10, 6))
    df["magnitude1"].plot(style="ro-")
    plt.title("Magnitude")
    plt.ylabel("Magnitude")
    plt.xticks(rotation=15)
    os.makedirs("results/large_earthquakes_analysis", exist_ok=True)
    plt.savefig("results/large_earthquakes_analysis/EQ_with_mag_6_graph.png")
    plt.clf()
    # plot on map
    df = df[df["district_id"] < 9]
    df = df[["latitude", "longitude", "magnitude1"]]
    df = df.sort_values("magnitude1").rename(columns={"magnitude1": "magnitude"})
    fig = px.scatter_mapbox(
        data_frame=df,
        lat="latitude",
        lon="longitude",
        color="magnitude",
        size="magnitude",
        size_max=8,
        opacity=0.9,
        zoom=3.5,
        height=700,
        width=900,
        color_continuous_scale="sunset",
    )
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.write_image("results/large_earthquakes_analysis/plot_on_map.png")
    print("Finished!")
