"""
Test Regressor without training (needs to be already trained obviously)
Loading model regressor.sav.
"""

import pickle
from common_functions import *
from tqdm import tqdm


# LOADING MODEL
regressor = pickle.load(open('regressor.sav', 'rb'))
coefficients = []
smp_list = []

# LOADING DATA
_, test_dl = load_data()

# TEST FEATURES EXTRACTION
for batch in tqdm(test_dl, total=len(test_dl.dataset), desc='test features extraction'):
    coeffs, smp = extract_regression_features(batch)
    coefficients.append(coeffs)
    smp_list.append(smp)

x_test = coefficients
y_test_re = smp_list

# PREDICTING
predictions = regressor.predict(x_test)

# EVALUATION METRICS
print_regression_evaluation_metrics(y_test_re, predictions)
