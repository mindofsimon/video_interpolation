"""
Test Regressor without training (needs to be already trained obviously)
Loading model regressor.sav.
"""

import pickle
from common_functions import *
import csv


regressor = pickle.load(open('regressor.sav', 'rb'))
coefficients = []
smp_list = []

# in csv_test_path put the path of the csv file containing testing video informations
csv_test_path = 'C:/Users/maria/OneDrive/Desktop/polimi/MAE/thesis/create_interpolated_videos/stamm_test_videos.csv'
with open(csv_test_path, 'r') as videos:
    data = csv.reader(videos, delimiter=',')
    for row in data:
        video_path = row[0]
        video_label = row[1]
        video_smp = row[2]
        # CREATING JSON FILES AND POPULATING LISTS
        create_json_files(video_path)
        frame_size_list, frame_type_list, color_list = read_json_file()

        # DECOMPOSING EFS TO FIND RESIDUALS
        residuals = efs_processing(frame_size_list, frame_type_list)

        # AUTOREGRESSIVE MODEL
        coefficients.append(build_ar_model(residuals))
        smp_list.append(video_smp)

x_test = coefficients
y_test_re = smp_list

# PREDICTING
predictions = regressor.predict(x_test)

# EVALUATION METRICS
# function receives real test labels and predictions
print_regression_evaluation_metrics(y_test_re, predictions)
