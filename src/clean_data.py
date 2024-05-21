import os
import sqlite3
import pandas as pd
from src.utils import *


def convert_raw_data_to_database():
    print("Starting to convert raw txt files to sql database ...")
    if os.path.exists("data/processed/earthquake.db"):
        print("You already have the database!")
        return
    data_path = "data/raw"
    year_month_sub = gen_year_month_list(1997, 10, 2022, 3)
    connection = sqlite3.connect("data/processed/earthquake.db")
    cursor = connection.cursor()
    query = """
        CREATE TABLE earthquake (
            datetime TEXT, latitude REAL, longitude REAL, depth REAL, magnitude1 REAL, magnitude2 REAL,
            max_intensity REAL, district_id INTEGER, region_id INTEGER, description TEXT
        )
        """
    cursor.execute(query)
    for yms in year_month_sub:
        dl_filename = data_path + "/h" + yms + ".txt"
        day = 1
        with open(dl_filename) as f:
            for line in f:
                if line.strip()[0:1] in "0123456789" and line.strip()[0:1] != "":
                    if line[2:4] != "  ":
                        day = int(line[2:4].strip())
                    time = line[5:13].replace(" ", ":")
                    dt = yms[:4] + "-" + yms[4:] + f"-{day:02} " + time
                    lat = float(line[21:23]) + float(line[24:28]) / 60
                    lot = float(line[34:37]) + float(line[38:42]) / 60
                    dep = float(line[48:51])
                    if line[55:58].strip() == "":
                        mag1 = None
                    else:
                        mag1 = float(line[55:58])
                    if line[60:63].strip() == "":
                        mag2 = None
                    else:
                        mag2 = float(line[60:63])
                    if line[65:66].strip() == "":
                        its = None
                    elif line[65:66].strip() == "A":
                        its = 5
                    elif line[65:66].strip() == "B":
                        its = 5.5
                    elif line[65:66].strip() == "C":
                        its = 6
                    elif line[65:66].strip() == "D":
                        its = 6.5
                    else:
                        its = float(line[65:66])
                    did = int(line[68:69])
                    if line[70:73].strip() == "":
                        rid = None
                    else:
                        rid = int(line[70:73])
                    dec = line[75:].strip()
                    query = f"""
                        INSERT INTO earthquake(
                            datetime, latitude, longitude, depth, magnitude1, magnitude2, max_intensity,
                            district_id, region_id, description
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """
                    cursor.execute(
                        query, (dt, lat, lot, dep, mag1, mag2, its, did, rid, dec)
                    )
        connection.commit()
    connection.close()
    print("Finished!")


def pre_process_data():
    print("Starting to pre-process data...")
    # Loading data from sql to dataframe
    cnx = sqlite3.connect("data/processed/earthquake.db")
    df = pd.read_sql_query("SELECT * FROM earthquake", cnx)
    # Fixing the time column datatype
    df["time"] = pd.to_datetime(df["datetime"])
    # Sorting the dataframe with respect to time
    df = df.sort_values(by="time")
    # Converting to time-series with respect to "time" column
    df.set_index("time", inplace=True)
    # Deleting the duplicate values
    df = df[~df.index.duplicated(keep="first")]
    df.to_csv("data/processed/time_series_data.csv")
    print("Finished!")
