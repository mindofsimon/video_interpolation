"""
Regressor able to detect speed manipulation parameter of a manipulated video.
Original videos labelled with 1.
Manipulated videos labelled with speed manipulation parameter (regression label).
Train and Test video informations (path, classification label, speed manipulation parameter) are stored in csv files,
be careful to set the appropriate path in 'csv_train_path' and 'csv_test_path' variables.
Needs common_functions.py to implement all the fundamental processing functions.
"""

from sklearn.svm import SVR
import pickle
from common_functions import *  # function used both by classification and regression scripts
import csv
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold


# MAIN PROGRAM
frame_size_list = []  # frame sizes
frame_type_list = []  # frame types
color_list = []  # useful just for plotting efs
x_train = []  # ar coefficients of training videos
y_train_re = []  # training labels for regression
x_test = []  # ar coefficients of testing videos
y_test_re = []  # testing labels for regression (speed manipulation parameter estimation)
coefficients = []  # coefficients of the ar model modeling the residual sequence
smp_list = []  # speed manipulation parameters of the videos

# **************************REGRESSION**************************
# in csv_train_path put the path of the csv file containing training video informations
csv_train_path = 'C:/Users/maria/OneDrive/Desktop/polimi/MAE/thesis/create_interpolated_videos/stamm_train_videos.csv'
with open(csv_train_path, 'r') as videos:
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

x_train = coefficients
y_train_re = smp_list
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


# COMPLETE DATASET (to use in cross validation)
x = x_train + x_test
y_re = y_train_re + y_test_re


# BUILING SVR (SPEED MANIPULATION PARAMETER ESTIMATION)
regressor = SVR(kernel='rbf')

# TRAINING SVR
regressor.fit(x_train, y_train_re)

# SAVING MODEL
model_name = "regressor.sav"
remove_previous_file(model_name)
pickle.dump(regressor, open(model_name, 'wb'))

# PREDICTING
predictions = regressor.predict(x_test)

# EVALUATION METRICS
# function receives real test labels and predictions
print_regression_evaluation_metrics(y_test_re, predictions)
cv = KFold(n_splits=10, random_state=1, shuffle=True)
cv_predictions = cross_val_predict(regressor, x, y_re, cv=cv, n_jobs=-1)
mean_bias_cv, variance_cv = regression_error_metrics(cv_predictions, y_re)
print("Cross Validation Mean Bias: " + str(mean_bias_cv))
print("Cross Validation Variance: " + str(variance_cv))
