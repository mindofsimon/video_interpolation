"""
Test Classifier without training (needs to be already trained obviously)
Loading model classifier.sav.
"""

import pickle
from common_functions import *
import csv


classifier = pickle.load(open('classifier.sav', 'rb'))
coefficients = []
label_list = []


# in csv_test_path put the path of the csv file containing testing video informations
csv_test_path = '/nas/home/smariani/video_interpolation/stamm/test.csv'
with open(csv_test_path, 'r') as videos:
    data = csv.reader(videos, delimiter=',')
    for row in data:
        video_path = row[0]
        video_label = row[1]
        # CREATING JSON FILES AND POPULATING LISTS
        create_json_file(video_path)
        frame_size_list, frame_type_list, color_list = read_json_file()

        # DECOMPOSING EFS TO FIND RESIDUALS
        residuals = efs_processing(frame_size_list, frame_type_list)

        # AUTOREGRESSIVE MODEL
        coefficients.append(build_ar_model(residuals))
        label_list.append(video_label)

x_test = coefficients
y_test_cl = label_list

# PREDICTING
predictions = classifier.predict(x_test)

# EVALUATION METRICS
# function receives real test labels and predictions
print_classification_evaluation_metrics(y_test_cl, predictions)
