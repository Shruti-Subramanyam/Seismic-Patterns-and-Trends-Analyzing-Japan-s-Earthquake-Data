[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/h_LXMCrc)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=12800422&assignment_repo_type=AssignmentRepo)

# DSCI 510 Final Project

## Name of the Project

Seismic Patterns and Trends: Analyzing Japan’s Earthquake Data

## Team Members (Name and Student IDs)

Kyosuke Chikamatsu(1485885653), Shruti Subramanyam(3541420114)

## Instructions to create a conda environment

In order to run this project, you need to create and activate a specific conda environment. You can run:

`conda env create -f environment.yml`  
`conda activate finalpj`

If you want to delete the environment, you can run:

`conda remove -n finalpj --all`

## Instructions on how to install the required libraries

In order to run this project, you need to have some modules installed in your Anaconda environment. You can run:

`pip install -r requirements.txt`

The above command should be executed after activating `finalpj`.

## Instructions on how to download the data

In order to download data, you need to run `main.py`. You can run:

`python main.py`

If you just want to run the data download section, you can call a Python interpreter and run:

`from src.get_data import *`  
`download_historical_data()`  
`get_region_master()`

These functions use internet connection, so check your environment in advance.  
You will get the download data in `data/raw` and `data/processed` directory.

## Instructions on how to clean the data

In order to clean the data, you need to run `main.py`. You can run:

`python main.py`

If you just want to run the data cleaning section, you can call a Python interpreter and run:

`from src.clean_data import *`  
`convert_raw_data_to_database()`  
`pre_process_data()`

The first function uses `data/raw` directory for input. The second function uses `data/processed/earthquake.db` for
input, but we do not share this data due to its size. So you need to run the first function
`convert_raw_data_to_database()` OR download processed data from `data/processed/earthquake.db.url` in advance to run
the second function.  
You will get the processed data in `data/processed` directory.

## Instructions on how to run analysis code

In order to run analysis, you need to run `main.py`. You can run:

`python main.py`

If you just want to run the analysis section, you can call a Python interpreter and run:

`from src.run_analysis import *`  
`large_earthquakes()`  
`analysis_by_district()`  
`lstm() # This function actually not only analyze data but also visualize the results`

These functions use `data/processed` directory for input, but we do not share some processed data due to their size.
So you need to run the data download and cleaning section OR download processed data from
`data/processed/earthquake.db.url` and `data/processed/time_series_data.csv.url` in advance to run these functions.   
You will get analyzed data in `data/analyzed` directory and lstm analysis graphs in `results/lstm_analysis` directory.

## Instructions on how to create visualizations

In order to create visualizations, you need to run `main.py`. You can run:

`python main.py`

If you just want to run the visualization section, you can call a Python interpreter and run:

`from src.visualize_results import *`  
`plot_vars_distribution()`  
`data_for_various_plots()`  
`plot_large_earthquakes()`  
`visualize_analysis_by_district()`

These functions use `data/processed` and `data/analyzed` directory for input.  
You will get visualized graphs in `results/exploratory_data_analysis`, `results/large_earthquakes_analysis`
and `results/analysis_by_district` directories respectively.