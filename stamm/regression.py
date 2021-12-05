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
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from tqdm import tqdm


frame_size_list = []  # frame sizes
frame_type_list = []  # frame types
color_list = []  # useful just for plotting efs
x_train = []  # ar coefficients of training videos
y_train_re = []  # training labels for regression
x_test = []  # ar coefficients of testing videos
y_test_re = []  # testing labels for regression (speed manipulation parameter estimation)
coefficients = []  # coefficients of the ar model modeling the residual sequence
smp_list = []  # speed manipulation parameters of the videos


# LOADING DATA
train_dl, test_dl = load_data()

# TRAIN FEATURES EXTRACTION
for batch in tqdm(train_dl, total=len(train_dl.dataset), desc='train features extraction'):
    coeffs, smp = extract_regression_features(batch)
    coefficients.append(coeffs)
    smp_list.append(smp)

x_train = coefficients
y_train_re = smp_list
coefficients = []
smp_list = []

# TEST FEATURES EXTRACTION
for batch in tqdm(test_dl, total=len(test_dl.dataset), desc='test features extraction'):
    coeffs, smp = extract_regression_features(batch)
    coefficients.append(coeffs)
    smp_list.append(smp)

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
print_regression_evaluation_metrics(y_test_re, predictions)
cv = KFold(n_splits=10, random_state=1, shuffle=True)
cv_predictions = cross_val_predict(regressor, x, y_re, cv=cv, n_jobs=-1)
mean_bias_cv, variance_cv = regression_error_metrics(cv_predictions, y_re)
print("Cross Validation Mean Bias: " + str(mean_bias_cv))
print("Cross Validation Variance: " + str(variance_cv))
