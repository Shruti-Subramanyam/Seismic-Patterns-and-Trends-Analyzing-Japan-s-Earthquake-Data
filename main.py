from src.get_data import *
from src.clean_data import *
from src.run_analysis import *
from src.visualize_results import *

# get data
download_historical_data()
get_region_master()

# clean data
convert_raw_data_to_database()
pre_process_data()

# do exploratory data analysis and visualization
plot_vars_distribution()
data_for_various_plots()

# analyze large earthquakes and visualize them
large_earthquakes()
plot_large_earthquakes()

# analyze data by district and visualize the results
analysis_by_district()
visualize_analysis_by_district()

# develop prediction model and visualize the results
lstm()
