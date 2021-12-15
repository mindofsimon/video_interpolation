"""
Test Classifier without training (needs to be already trained obviously)
Loading model classifier.sav.
"""

import pickle
from common_functions import *
from tqdm import tqdm


# LOADING MODEL
classifier = pickle.load(open('classifier_15_new.sav', 'rb'))
coefficients = []
label_list = []

# LOADING DATA
_, test_dl = load_data()

# TEST FEATURES EXTRACTION
print("FEATURE EXTRACTION")
for batch in tqdm(test_dl, total=len(test_dl.dataset), desc='test features extraction'):
    coeffs, label = extract_classification_features(batch)
    coefficients.append(coeffs)
    label_list.append(label)

x_test = coefficients
y_test_cl = label_list

# PREDICTING
predictions = classifier.predict(x_test)

# EVALUATION METRICS
print_classification_evaluation_metrics(y_test_cl, predictions)
