import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import zipfile
import io
from src.utils import *


def get_region_master():
    print("Starting to get region master ...")
    output_filepath = "data/processed/region_master.csv"
    if os.path.exists(output_filepath):
        print("You already have the region master!")
        return
    pre_url = "https://www.data.jma.go.jp/eqev/data/bulletin/catalog/appendix/regname"
    suf_url = "_e.html"
    url_list = [pre_url + str(i) + suf_url for i in range(1, 9)]
    # url = url_list[0]
    df_list = []
    for url in url_list:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        district_name = soup.css.select("#contents > table")[0].find("caption").text
        df = pd.read_html(url)[0]
        df.insert(1, "DISTRICT NAME", district_name)
        df_list.append(df)
    pd.concat(df_list, axis=0, ignore_index=True).to_csv(output_filepath, index=False)
    print("Finished!")


def download_historical_data():
    print("Starting to download historical data ...")
    extract_path = "data/raw"
    year_month_sub = gen_year_month_list(1997, 10, 2022, 3)
    for yms in year_month_sub:
        dl_filename = extract_path + "/h" + yms + ".txt"
        if os.path.exists(dl_filename):
            print("\r" + dl_filename + " already exists", end="")
            continue
        url = (
            "https://www.data.jma.go.jp/eqev/data/bulletin/catalog/table2/h"
            + yms
            + "t.zip"
        )
        response = requests.get(url)
        print("\r" + "Downloading " + dl_filename + " ...", end="")
        with zipfile.ZipFile(io.BytesIO(response.content), "r") as zip_ref:
            zip_ref.extractall(extract_path)
    print("\nFinished!")
