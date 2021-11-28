"""
Classifier to disriminate between original and manipulated(interpolation) videos.
Original videos labelled with 0.
Manipulated videos labelled with 1.
Train and Test video informations (path, classification label, speed manipulation parameter) are stored in csv files,
be careful to set the appropriate path in 'csv_train_path' and 'csv_test_path' variables.
Needs common_functions.py to implement all the fundamental processing functions.
"""

from sklearn import svm
import pickle
from common_functions import *  # function used both by classification and regression scripts
import csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# MAIN PROGRAM
frame_size_list = []  # frame sizes
frame_type_list = []  # frame types
color_list = []  # useful just for plotting efs
x_train = []  # ar coefficients of training videos
y_train_cl = []  # training labels for binary classification (speed manipulation detection)
x_test = []  # ar coefficients of testing videos
y_test_cl = []  # testing labels for binary classification (speed manipulation detection)
coefficients = []  # coefficients of the ar model modeling the residual sequence
label_list = []  # list of video labels


# in csv_train_path put the path of the csv file containing training video informations
csv_train_path = 'C:/Users/maria/OneDrive/Desktop/polimi/MAE/thesis/create_interpolated_videos/stamm_train_videos.csv'
with open(csv_train_path, 'r') as videos:
    data = csv.reader(videos, delimiter=',')
    for row in data:
        video_path = row[0]
        video_label = row[1]

        # CREATING JSON FILES AND POPULATING LISTS
        create_json_files(video_path)
        frame_size_list, frame_type_list, color_list = read_json_file()

        # DECOMPOSING EFS TO FIND RESIDUALS
        residuals = efs_processing(frame_size_list, frame_type_list)

        # AUTOREGRESSIVE MODEL
        coefficients.append(build_ar_model(residuals))
        label_list.append(video_label)

x_train = coefficients
y_train_cl = label_list
coefficients = []
label_list = []

# in csv_test_path put the path of the csv file containing testing video informations
csv_test_path = 'C:/Users/maria/OneDrive/Desktop/polimi/MAE/thesis/create_interpolated_videos/stamm_test_videos.csv'
with open(csv_test_path, 'r') as videos:
    data = csv.reader(videos, delimiter=',')
    for row in data:
        video_path = row[0]
        video_label = row[1]
        # CREATING JSON FILES AND POPULATING LISTS
        create_json_files(video_path)
        frame_size_list, frame_type_list, color_list = read_json_file()

        # DECOMPOSING EFS TO FIND RESIDUALS
        residuals = efs_processing(frame_size_list, frame_type_list)

        # AUTOREGRESSIVE MODEL
        coefficients.append(build_ar_model(residuals))
        label_list.append(video_label)

x_test = coefficients
y_test_cl = label_list

# COMPLETE DATASET (to use in cross validation)
x = x_train + x_test
y_cl = y_train_cl + y_test_cl

# BUILING SVM (SPEED MANIPULATION DETECTION)
classifier = svm.SVC(kernel='rbf')

# TRAINING SVM
classifier.fit(x_train, y_train_cl)

# SAVING MODEL
model_name = "classifier.sav"
remove_previous_file(model_name)
pickle.dump(classifier, open(model_name, 'wb'))

# PREDICTING
predictions = classifier.predict(x_test)

# EVALUATION METRICS
# function receives real test labels and predictions
print_classification_evaluation_metrics(y_test_cl, predictions)
cv = KFold(n_splits=10, random_state=1, shuffle=True)
scores = cross_val_score(classifier, x, y_cl, scoring='accuracy', cv=cv, n_jobs=-1)
print("Cross Validation Accuracy: " + str((round(scores.mean(), 3)) * 100) + "%")
