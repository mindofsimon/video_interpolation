from sklearn import svm
import pickle
from global_interp_cnn_utils import *
import numpy as np


def train_svc():
    with open('scores/train_scores.pkl', 'rb') as f1:
        train_scores = pickle.load(f1)
    with open('scores/train_labels.pkl', 'rb') as f2:
        train_labels = pickle.load(f2)
    classifier = svm.SVC(kernel='rbf', probability=True)

    # TRAINING SVM
    print("FITTING THE MODEL...")
    classifier.fit(train_scores, train_labels)

    # SAVING MODEL
    model_name = "/nas/home/smariani/video_interpolation/interp_cnn/models/speednet_svc.sav"
    pickle.dump(classifier, open(model_name, 'wb'))


def test_svc():
    with open('scores/test_scores.pkl', 'rb') as f1:
        test_scores = pickle.load(f1)
    with open('scores/test_labels.pkl', 'rb') as f2:
        test_labels = pickle.load(f2)
    classifier = pickle.load(open('/nas/home/smariani/video_interpolation/interp_cnn/models/speednet_svc.sav', 'rb'))
    predictions = classifier.predict(test_scores)
    predictions = [float(x) for x in predictions]
    predictions = np.array(predictions)
    test_labels = np.array(test_labels)
    true_pos = np.sum(predictions == test_labels)
    print_eval_metrics(test_labels, predictions, true_pos, len(predictions))


if __name__ == '__main__':
    train_svc()
    test_svc()