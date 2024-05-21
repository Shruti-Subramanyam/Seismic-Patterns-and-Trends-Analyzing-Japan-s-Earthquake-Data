import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sqlite3
from sklearn.preprocessing import RobustScaler
import keras
from keras.layers import Masking

pd.options.mode.chained_assignment = None


# Analysis to find the largest earthquake until now
def large_earthquakes():
    print("Largest earthquake analysis..")
    # Reading the csv file
    file_path = "data/processed/time_series_data.csv"
    df = pd.read_csv(file_path, index_col=0)
    df["time"] = df.index
    df["magnitude1"] = pd.to_numeric(df["magnitude1"], errors="coerce")
    df_eq_large = df[df["magnitude1"] > 6].copy()
    df_eq_large["time"] = pd.to_datetime(df_eq_large["time"])
    df_eq_large["time_diff_day"] = df_eq_large["time"].diff()
    print(
        f"There are {len(df_eq_large)} earthquakes greater than magnitude of 6 in the last 25years"
    )
    print(
        "Largest earthquake in Japan until now is:\n"
        f"{df_eq_large.loc[df_eq_large.magnitude1 == df_eq_large.magnitude1.max(), 'latitude':'magnitude1']}"
    )
    save_path = "data/analyzed/time_series_data_lt6.csv"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_eq_large.to_csv(save_path, index=False)
    print("Finished!")


def create_dataset(x, y, time_steps=1):
    xs, ys = [], []
    for i in range(len(x) - time_steps):
        v = x.iloc[i : (i + time_steps)].to_numpy()
        xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(xs), np.array(ys)


def lstm():
    print("Starting to develop LSTM model...")
    # Reading the csv file
    file_path = "data/processed/time_series_data.csv"
    df = pd.read_csv(file_path, index_col=0)
    df["time"] = df.index
    df["magnitude1"] = pd.to_numeric(df["magnitude1"], errors="coerce")
    df_eq = df[["time", "magnitude1", "longitude", "latitude", "depth"]]
    df_eq = df_eq[df_eq["magnitude1"] > 4].copy()
    df_eq["time"] = pd.to_datetime(df_eq["time"])
    df_eq["timestamps"] = df_eq["time"]

    # Split the data into train/test with 10% of data as test dataset
    df_eq["time_diff"] = df_eq["timestamps"].diff()
    df_eq["time_diff_float"] = df_eq["time_diff"].apply(lambda x: x.total_seconds())
    df_eq["mag_roll_10"] = df_eq["magnitude1"].rolling(window=10).mean()
    df_eq.dropna(inplace=True)
    df_eq_model = df_eq.drop(columns=["timestamps", "time_diff", "time"])
    train_size = int(len(df_eq_model) * 0.9)
    train, test = (
        df_eq_model.iloc[0:train_size],
        df_eq_model.iloc[train_size : len(df_eq_model)],
    )
    print(train.shape, test.shape)
    # Scaling the data
    f_columns = ["longitude", "latitude", "depth", "time_diff_float", "mag_roll_10"]
    f_transformer = RobustScaler()
    mag_transformer = RobustScaler()
    time_diff_float_transformer = RobustScaler()
    f_transformer = f_transformer.fit(train[f_columns].to_numpy())
    mag_transformer = mag_transformer.fit(train[["magnitude1"]])
    time_diff_float_transformer = time_diff_float_transformer.fit(
        train[["time_diff_float"]]
    )
    train.loc[:, f_columns] = f_transformer.transform(train[f_columns].to_numpy())
    train["magnitude1"] = mag_transformer.transform(train[["magnitude1"]])
    test.loc[:, f_columns] = f_transformer.transform(test[f_columns].to_numpy())
    test["magnitude1"] = mag_transformer.transform(test[["magnitude1"]])
    # Preparing data for lstm
    time_steps = 60  # 60
    print("Preparing data for lstm..")
    x_train, y_train = create_dataset(train, train["magnitude1"], time_steps=time_steps)
    x_test, y_test = create_dataset(test, test["magnitude1"], time_steps=time_steps)
    print("Finished!")
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    # Model architecture
    model = keras.Sequential()
    # Adding mask layer for NaN values
    model.add(
        Masking(mask_value=-1.0, input_shape=(x_train.shape[1], x_train.shape[2]))
    )
    # Adding bidirectional layer

    model.add(
        keras.layers.Bidirectional(
            keras.layers.LSTM(
                units=128, input_shape=(x_train.shape[1], x_train.shape[2])
            )
        )
    )
    # Adding dropout layer to regularize complexities
    model.add(keras.layers.Dropout(rate=0.2))
    # Add output layer
    model.add(keras.layers.Dense(units=1))
    # Compiling the model
    model.compile(loss="mean_squared_error", optimizer="adam")
    history = model.fit(
        x_train,
        y_train,
        epochs=10,  # 10
        batch_size=32,
        validation_split=0.2,
        shuffle=False,  # As it is time-series
    )
    print("Plotting results...")
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.legend()
    plt.title("Train Vs Test")
    if not os.path.exists("results/lstm_analysis"):
        os.makedirs("results/lstm_analysis")
    plt.savefig("results/lstm_analysis/EQ_lstm_train_test_graph.png")
    plt.clf()
    y_pred = model.predict(x_test)
    y_test_inv = time_diff_float_transformer.inverse_transform(y_test.reshape(1, -1))
    y_pred_inv = time_diff_float_transformer.inverse_transform(y_pred)
    plt.figure(figsize=(14, 10))
    plt.title("Train Vs Test")
    plt.plot(y_test_inv.flatten(), marker=".", label="True")
    plt.plot(y_pred_inv.flatten(), marker=".", label="Predicted")
    plt.legend()
    plt.savefig("results/lstm_analysis/EQ_lstm_pred_graph.png")
    plt.clf()
    print("Finished!")


def analysis_by_district():
    print("Starting to analyze data by district...")
    # Loading data from sql
    cnx = sqlite3.connect("data/processed/earthquake.db")
    ft = pd.read_sql_query(
        """SELECT district_id, region_id, COUNT(*) FROM earthquake WHERE magnitude1 > 6 AND region_id IS NOT NULL 
        GROUP BY district_id, region_id""",
        cnx,
    )
    # Loading region master data
    rm = pd.read_csv("data/processed/region_master.csv")
    rm.columns = ["district_id", "district_name", "region_id", "region_name"]
    # Making district master data
    rm_by_district = rm[["district_id", "district_name"]].drop_duplicates()
    # making frequency table for each region
    ftm = pd.merge(ft, rm, how="inner", on=["district_id", "region_id"])
    ftm.reindex(columns=["district_name", "region_name", "COUNT(*)"]).rename(
        columns={"COUNT(*)": "count"}
    ).to_csv("data/analyzed/frequency_table.csv", index=False)
    # loading monthly data by district
    ft_by_date = pd.read_sql_query(
        """SELECT date, sum(magnitude1) as magnitude, district_id 
        FROM (SELECT strftime('%Y-%m', datetime) as date, magnitude1, district_id FROM earthquake) 
        GROUP BY date, district_id
        ORDER By date, district_id""",
        cnx,
    )
    # text to date type
    ft_by_date["date"] = pd.to_datetime(ft_by_date["date"], format="%Y-%m")
    # merge with district name
    pd.merge(ft_by_date, rm_by_district, on=["district_id"], how="inner").pivot(
        index="date", columns="district_name", values="magnitude"
    ).cumsum().to_csv("data/analyzed/monthly_magnitude_by_district.csv")
    cnx.close()
    print("Finished!")
