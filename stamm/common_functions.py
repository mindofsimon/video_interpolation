"""
This script contains functions used both by classification and regression scripts.
Some of them are related to the construction of the final training features starting from the video frames.
Some others are just utils, like rounding decimals or printing output scores.
"""

import json
import numpy as np
import os
import scipy.signal
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from load_data import VideoDataset
from torch.utils.data import DataLoader


def load_data():
    """
    Loading data from csv files (path, label, speed manipulation parameter) to VideoDataset class.
    Then from VideoDataset class into DataLoaders.

    All videos (train/test) are loaded in single batch.

    Be careful to put in originals csv the path to the video info csv files!

    :return: two data loaders (one for train videos and one for test videos).
    """

    # train ds
    videos_csv = '/nas/home/smariani/video_interpolation/datasets/dfdc/partition35/train/train.csv'
    dataset = VideoDataset(videos_csv)
    train_data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    # test ds
    videos_csv = '/nas/home/smariani/video_interpolation/datasets/dfdc/partition35/test/test.csv'
    test_dataset = VideoDataset(videos_csv)
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=2)

    return train_data_loader, test_data_loader


def print_regression_evaluation_metrics(y_test_re, predictions):
    """
    Printing regression estimations, mean bias and variance
    :param y_test_re: test regression labels (1 original, >1 sped up video, <1 slowed down video)
    :param predictions: test predictions
    :return:
    """
    mean_bias, variance = regression_error_metrics(predictions, y_test_re)
    predictions = cut_list_decimals(predictions)  # just for the print
    print("\n*********************REGRESSION*********************")
    print("Real Speed Manipulation Parameters: " + str(y_test_re))
    print("Estimated Speed Manipulation Parameters: " + str(predictions))
    print("Mean Bias Of Estimation (NO CV): " + str(mean_bias))
    print("Variance Of Estimation (NO CV): " + str(variance))


def print_classification_evaluation_metrics(y_test_cl, predictions):
    """
    Printing classification accuracy and confusion matrix
    :param y_test_cl: test classification labels (0 original, 1 manipulated)
    :param predictions: test predictions
    :return: nothing
    """
    conf_matrix = confusion_matrix(y_test_cl, predictions, labels=["0", "1"])
    tn, fn, fp, tp = conf_matrix.ravel()
    plt.figure(figsize=(10, 10), dpi=300)
    df_cm = pd.DataFrame(conf_matrix, index=["original", "modified"], columns=["original", "modified"])
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix (GLOBAL CLASSIFICATION)")
    plt.savefig("cm_classification.png")
    plt.show()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    print("\n*********************GLOBAL CLASSIFICATION*********************")
    print("Classification Accuracy (NO CV): " + str(round(accuracy * 100, 3)) + "%")


def remove_previous_file(filename):
    """
    Remove specified file
    :param filename: file to remove
    :return: nothing
    """
    if filename in os.listdir():
        os.remove(filename)


def cut_list_decimals(number_list):
    """
    Cutting decimals from all elements of a list
    :param number_list: input list (of numbers...)
    :return: list of elements cut at 3rd decimal
    """
    number_list = np.array(number_list)
    number_list = np.round(number_list, 3)

    return number_list


def regression_error_metrics(pred, true_lab):
    """
    Computing error metrics for the regression case.
    :param pred: predictions list
    :param true_lab: true labels list
    :return: mean bias and variance
    """
    pred = np.array(pred)
    true_lab = np.array(true_lab)
    true_lab = true_lab.astype(float)
    m_b = pred - true_lab
    var = np.var(m_b)
    m_b = sum(m_b)/len(pred)
    return m_b, var


def efs_processing(f_size_list, f_type_list):
    """
    Computes the efs sequence and its estimation starting from video frames types and sizes.
    The estimation is obtained through decomposition of efs, median filtering and recomposition.
    It returns the residual sequence (efs - estimation).
    If the residuals sequence is close to 0 the video is likely to be an original one.
    If the residuals sequence is noisy the video is likely to be a manipulated one.
    :param f_size_list: frame size list of a video
    :param f_type_list: frame type list of a video
    :return: residual sequence (difference between efs sequence and its estimation).
    """
    i_frames_list, i_frames_map, b_frames_list, b_frames_map, p_frames_list, p_frames_map = get_frame_type_lists(f_type_list, f_size_list)
    med_i, med_b, med_p = median_filtering(i_frames_list, b_frames_list, p_frames_list)
    original_efs_estimation = recombine_efs_estimations(med_i, med_b, med_p, i_frames_map, b_frames_map, p_frames_map, f_size_list)
    residuals_sequence = calculate_residuals(f_size_list, original_efs_estimation)
    return residuals_sequence


def median_filtering(i_frames, b_frames, p_frames):
    """
    Median filtering the efs subsequences
    :param i_frames: I frames sequence
    :param b_frames: B frames sequence
    :param p_frames: P frames sequence
    :return: the three median filtered sequences
    """
    med_i_frames = scipy.signal.medfilt(i_frames)
    med_b_frames = scipy.signal.medfilt(b_frames)
    med_p_frames = scipy.signal.medfilt(p_frames)
    return med_i_frames, med_b_frames, med_p_frames


def build_ar_model(res):
    """
    Building an AutoRegressive model onto the residual sequence
    :param res: residuals (difference between efs sequence and its estimation)
    :return: coefficients of the AR model
    """
    ar_residuals = AutoReg(res, 21, "n").fit()  # order of the model can go from 4 to 30 (STAMM)
    coeff = list(ar_residuals.params)
    return coeff


def calculate_residuals(efs, estimation):
    """
    Simply subtracting two lists elementwise.
    :param efs: encoded frame size sequence of a video
    :param estimation: encoded frame size sequence estimation of a video
    :return: residuals of the difference between the two sequences.
    """
    difference = np.array(efs) - np.array(estimation)

    return difference


def recombine_efs_estimations(medi, medb, medp, mapi, mapb, mapp, size_list):
    """
    Recombining EFS sequence starting from subsequences and mappings.
    :param medi: median filtered I frames sequence
    :param medb: median filtered B frames sequence
    :param medp: median filtered P frames sequence
    :param mapi: mapping of I frames to the original sequence
    :param mapb: mapping of B frames to the original sequence
    :param mapp: mapping of P frames to the original sequence
    :param size_list: frame size list of the video
    :return: recombined encoded frame size sequence (it will represent an estimation of it)
    """
    n_elements = len(size_list)
    estimation = [None]*n_elements
    estimation = np.array(estimation)
    medi = np.array(medi)
    medb = np.array(medb)
    medp = np.array(medp)
    estimation[mapi] = medi
    estimation[mapb] = medb
    estimation[mapp] = medp

    return estimation


def get_frame_type_lists(type_list, size_list):
    """
    Decomposes encoded frame size sequence into subsequences (one for each frame type).
    It also returns the mapping in a way that we are able reconstruct the original sequence.
    :param type_list: list of frame types of a video
    :param size_list: list of frame sizes of a video
    :return: a decomposition of the video frame sequence (into frame types) and a mapping to reconstruct it later.
             For example, if we have the following inputs:
             type_list = [I,P,B,I,P,P,B,I,P,B]
             size_list = [4,2,2,5,2,1,2,5,2,3]
             As output we will get:
                the sizes sub lists (size of each frame):
                i_list = [4,5,5]
                b_list = [2,2,3]
                p_list = [2,2,1,2]
                and the mapping lists (index of each frame in the original sequence):
                i_mapping_lists = [0,3,7]
                b_mapping_lists = [2,6,9]
                p_mapping_lists = [1,4,5,8]
    """
    size_list = np.array(size_list)
    type_list = np.array(type_list)
    i_mapping_list = np.where(type_list == 'I')[0]
    i_list = size_list[i_mapping_list]
    b_mapping_list = np.where(type_list == 'B')[0]
    b_list = size_list[b_mapping_list]
    p_mapping_list = np.where(type_list == 'P')[0]
    p_list = size_list[p_mapping_list]

    return i_list, i_mapping_list, b_list, b_mapping_list, p_list, p_mapping_list


def create_json_file(vid):
    """
    Creates data.json containing video informations.
    :param vid: input video
    :return: json file (data.json) containing informations about video frames (frame size, frame type)
    """
    remove_previous_file('data.json')
    os.system('ffprobe -v error -hide_banner -of default=noprint_wrappers=0 -print_format json -select_streams v:0 -show_entries frame=pict_type,pkt_size ' + vid + ' > data.json')


def read_json_file():
    """
    Reads data.json and creates python lists containing video informations.
    :return: three lists scanning the json file (data.json) created for a video
             the lists are frame size, frame type and frame color.
             In the frame color one a color(Green/Red/Orange) is assigned
             to the three possible type of frames (I/P/B).
    """
    f_size_list = []   # to be filled with frame sizes in bytes
    f_type_list = []   # to be filled with frame types (I/B/P)
    c_list = []   # just for graphical reasons (I=green/B=red/P=orange)
    file = open('data.json')
    data = json.load(file)
    for frame in data['frames']:
        f_size_list.append(int(frame['pkt_size']))
        f_type_list.append(frame['pict_type'])
        if frame['pict_type'] == 'I':
            c_list.append('green')
        else:
            if frame['pict_type'] == 'B':
                c_list.append('red')
            else:
                c_list.append('orange')

    file.close()
    os.remove('data.json')
    return f_size_list, f_type_list, c_list


def extract_classification_features(batch):
    """
    Extracting coefficients (features) and label from a video (contained in batch: path, label, smp)
    :param batch: batch containing: video path, video label (0: original, 1: manipulated), video smp (speed manipulation parameter).
                  if video is original, smp is set to 1.
    :return: coefficients (features representing the video) and video label
    """

    video_path, video_label, _ = batch
    video_path = video_path[0]
    video_label = video_label[0]

    # CREATING JSON FILES AND POPULATING LISTS
    create_json_file(video_path)
    frame_size_list, frame_type_list, color_list = read_json_file()

    # DECOMPOSING EFS TO FIND RESIDUALS
    residuals = efs_processing(frame_size_list, frame_type_list)

    # AUTOREGRESSIVE MODEL
    coefficients = build_ar_model(residuals)

    return coefficients, video_label


def extract_regression_features(batch):
    """
        Extracting coefficients (features) and smp from a video (contained in batch: path, label, smp)
        :param batch: batch containing: video path, video label (0: original, 1: manipulated), video smp (speed manipulation parameter).
                      if video is original, smp is set to 1.
        :return: coefficients (features representing the video) and video smp
        """

    video_path, _, video_smp = batch
    video_path = video_path[0]
    video_smp = video_smp[0]

    # CREATING JSON FILES AND POPULATING LISTS
    create_json_file(video_path)
    frame_size_list, frame_type_list, color_list = read_json_file()

    # DECOMPOSING EFS TO FIND RESIDUALS
    residuals = efs_processing(frame_size_list, frame_type_list)

    # AUTOREGRESSIVE MODEL
    coefficients = build_ar_model(residuals)

    return coefficients, video_smp
