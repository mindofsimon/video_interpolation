"""
Utils for Interpolation CNN.
Including preprocessing steps for training/test/validation, data loading and output metrics.
"""
import os.path
import random
from torch.utils.data import DataLoader
from utils.load_data import VideoDataset
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
from utils.dense_optical_flow import of
from utils.naive_residuals import get_naive_residuals


def print_eval_metrics(test_labels, predictions_list, true_positives, total, net_type):
    """
    Printing confusion matrix and accuracy
    :param test_labels: test video labels
    :param predictions_list: predicted label
    :param true_positives: number of true positives
    :param total: total number of predictions
    :param net_type : name of the used model
    :return: nothing
    """
    # printing confusion matrix and accuracy
    conf_matrix = confusion_matrix(test_labels, predictions_list, labels=[0.0, 1.0])
    plt.figure(figsize=(10, 10), dpi=300)
    df_cm = pd.DataFrame(conf_matrix, index=["original", "modified"], columns=["original", "modified"])
    sn.heatmap(df_cm, annot=True, fmt='d', annot_kws={"size": 16})
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix (" + net_type.upper() + ")")
    plt.savefig('/nas/home/smariani/video_interpolation/interp_cnn/eval_metrics/cm_' + net_type + '.png')
    plt.show()
    accuracy = round(((true_positives / total) * 100), 3)
    print("Accuracy:" + str(accuracy) + "%")


def load_data(b_size):
    """
    Loading data from csv files (path) to VideoDataset class.
    Then from VideoDataset class into DataLoaders.

    All videos are loaded in single batch (containing: video path).
    Train/Test/Validation videos dataloaders will contain just original videos (interpolated video will be loaded at
    runtime, it has the same filename of the original but it's placed in a different folder).

    Be careful to put in originals csv the path to the video info csv files!

    :param b_size: how many elements to extract from the data loader at a time.
    :return: three data loaders (one for train videos, one for test videos and one for validation videos).
    """

    # train ds
    videos_csv = '/nas/home/smariani/video_interpolation/datasets/kinetics400/train.csv'
    dataset = VideoDataset(videos_csv)
    train_data_loader = DataLoader(dataset, batch_size=b_size, shuffle=True, num_workers=2)

    # test ds
    videos_csv = '/nas/home/smariani/video_interpolation/datasets/kinetics400/test.csv'
    test_dataset = VideoDataset(videos_csv)
    test_data_loader = DataLoader(test_dataset, batch_size=b_size, shuffle=True, num_workers=2)

    # valid ds
    videos_csv = '/nas/home/smariani/video_interpolation/datasets/kinetics400/validation.csv'
    valid_dataset = VideoDataset(videos_csv)
    valid_data_loader = DataLoader(valid_dataset, batch_size=b_size, shuffle=True, num_workers=2)

    return train_data_loader, test_data_loader, valid_data_loader


def train_data_processing(batch, t, c):
    """
    Data processing chain for train videos.
    From video path to input (to the model) tensor.
    :param batch: batch containing original video path
    :param t: temporal dimension (frame number)
    :param c: number of channels (for each frame)
    :return: input to the model(data, with batch size = BATCH_SIZE MODEL) and video labels.
             the input to the model are original and interpolated videos (t frames each of size n x n, processed with
             optical flow).
    """

    video_labels = []
    video_frames = []
    # spatial augmentation parameter
    n = random.randrange(64, 336)  # should be between 64 and 336 but it depends on gpu memory limit...
    # extracting data
    for v in batch:
        video_path = v
        video_labels.append([0.0])  # original
        video_labels.append([1.0])  # interpolated
        interpolated_path = "/nas/home/smariani/video_interpolation/datasets/kinetics400/minterpolate_60fps/train/" \
                            + os.path.basename(video_path)
        # optical flow (will also keep just t frames and resize them to n x n)
        original_of = get_naive_residuals(video_path, n, t, training=True)
        interp_of = get_naive_residuals(interpolated_path, n, t, training=True)
        video_frames.append(original_of)
        video_frames.append(interp_of)
    # building input tensor
    video_labels = torch.tensor(video_labels)
    data = generate_data(video_frames, n, t, c)
    return data, video_labels


def val_data_processing(batch, n, t, c):
    """
    Data processing chain for test and validation videos.
    :param batch: batch containing original video path
    :param n: spatial dimension (frame size)
    :param t: temporal dimension (frame number)
    :param c: number of channels (for each frame)
    :return: input to the model(data, with batch size = BATCH_SIZE_MODEL) and video labels.
             the input to the model are original and interpolated videos (t frames each of size n x n, processed with
             optical flow).
    """

    video_labels = []
    video_frames = []
    # extracting data
    for v in batch:
        video_path = v
        video_labels.append([0.0])  # original
        video_labels.append([1.0])  # interpolated
        interpolated_path = "/nas/home/smariani/video_interpolation/datasets/kinetics400/minterpolate_60fps/validation/" \
                            + os.path.basename(video_path)
        # optical flow (will also keep just t frames and resize them to n x n)
        original_of = get_naive_residuals(video_path, n, t, training=False)
        interp_of = get_naive_residuals(interpolated_path, n, t, training=False)
        video_frames.append(original_of)
        video_frames.append(interp_of)
    # building input tensor
    video_labels = torch.tensor(video_labels)
    data = generate_data(video_frames, n, t, c)
    return data, video_labels


def test_data_processing(batch, n, t, c):
    """
    Data processing chain for test  videos.
    :param batch: batch containing original video path
    :param n: spatial dimension (frame size)
    :param t: temporal dimension (frame number)
    :param c: number of channels (for each frame)
    :return: input to the model(data, with batch size = BATCH_SIZE_MODEL) and video labels.
             the input to the model are original and interpolated videos (t frames each of size n x n, processed with
             optical flow).
    """

    video_labels = []
    video_frames = []
    # extracting data
    for v in batch:
        video_path = v
        video_labels.append([0.0])  # original
        video_labels.append([1.0])  # interpolated
        interpolated_path = "/nas/home/smariani/video_interpolation/datasets/kinetics400/minterpolate_60fps/test/" \
                            + os.path.basename(video_path)
        # optical flow (will also keep just t frames and resize them to n x n)
        original_of = get_naive_residuals(video_path, n, t, training=False)
        interp_of = get_naive_residuals(interpolated_path, n, t, training=False)
        video_frames.append(original_of)
        video_frames.append(interp_of)
    # building input tensor
    video_labels = torch.tensor(video_labels)
    data = generate_data(video_frames, n, t, c)
    return data, video_labels


def generate_data(f_list, n, t, c):
    """
    Generate tensor as input for the model.
    Starts from the frames lists of the videos ([[original_vid1], [interp_vid1] , [original_vid2], [interp_vid2], ...]).
    Produces tensor of size [BATCH_SIZE_MODEL, N_CHANNELS(3), N_FRAMES(t), HEIGHT(n), WIDTH(n)]
    :param f_list: list of the preprocessed video frames (as descripted above)
    :param t: number of consecutive taken frames (temporal dimension)
    :param n: size of the frames (spatial dimension)
    :param c: number of channels (for each frame)
    :return: input tensor to feed the model and skip flag (if video has too low number of frames and has to be skipped)
    """

    frames_list = np.array(f_list)  # original, interpolated
    data = torch.autograd.Variable(torch.tensor(frames_list))
    data = torch.reshape(data, (len(f_list), t, n, n, c))
    data = torch.permute(data, [0, 4, 1, 2, 3])
    data = data.float()

    return data
