"""
Utils for SpeedNet.
Including preprocessing steps for training/test/validation, data loading and output metrics.
"""
import os.path
import random
import cv2
from torch.utils.data import DataLoader
from load_data import VideoDataset
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
from dense_optical_flow import of


def print_eval_metrics(test_labels, predictions_list, true_positives, total):
    """
    Printing confusion matrix and accuracy
    :param test_labels: test video labels
    :param predictions_list: predicted label
    :param true_positives: number of true positives
    :param total: total number of predictions
    :return: nothing
    """
    # printing confusion matrix and accuracy
    conf_matrix = confusion_matrix(test_labels, predictions_list, labels=[0.0, 1.0])
    plt.figure(figsize=(10, 10), dpi=300)
    df_cm = pd.DataFrame(conf_matrix, index=["original", "modified"], columns=["original", "modified"])
    sn.heatmap(df_cm, annot=True, fmt='d', annot_kws={"size": 16})
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix (SPEEDNET)")
    plt.savefig('/nas/home/smariani/video_interpolation/speednet/cm_speednet.png')
    plt.show()
    accuracy = round(((true_positives / total) * 100), 3)
    print("Accuracy:" + str(accuracy) + "%")


def load_data():
    """
    Loading data from csv files (path) to VideoDataset class.
    Then from VideoDataset class into DataLoaders.

    All videos are loaded in single batch (containing: video path).
    Train/Test/Validation videos dataloaders will contain just original videos (interpolated video will be loaded at
    runtime, it has the same filename of the original but it's placed in a different folder).

    Be careful to put in originals csv the path to the video info csv files!

    :return: three data loaders (one for train videos, one for test videos and one for validation videos).
    """

    # train ds
    videos_csv = '/nas/home/smariani/video_interpolation/datasets/kinetics400/train.csv'
    dataset = VideoDataset(videos_csv)
    train_data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    # test ds
    videos_csv = '/nas/home/smariani/video_interpolation/datasets/kinetics400/test.csv'
    test_dataset = VideoDataset(videos_csv)
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=2)

    # valid ds
    videos_csv = '/nas/home/smariani/video_interpolation/datasets/kinetics400/validation.csv'
    valid_dataset = VideoDataset(videos_csv)
    valid_data_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True, num_workers=2)

    return train_data_loader, test_data_loader, valid_data_loader


def center_crop(image):
    """
    Center crop (224 x 224) to apply to test and validation videos.
    :param image: input frame
    :return: cropped frame
    """

    center = [image.shape[0] / 2, image.shape[1] / 2]
    h = image.shape[0]
    w = 224
    x = int(center[1]) - int(w / 2)
    y = int(center[0]) - int(h / 2)

    crop_img = image[y:y + h, x:x + w]
    return crop_img


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    Resizing image keeping same ratio.
    :param image: image to resize
    :param width: desired width (height = None)
    :param height: desired height (width = None)
    :param inter: interpolation type (for the reconstruction)
    :return: resized image
    """

    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def train_data_processing(batch, t):
    """
    Data processing chain for train videos.
    From video path to input (to the model) tensor.
    :param batch: batch containing original video path
    :param t: temporal dimension (frame number)
    :return: input to the model(data, with batch size = 2), video labels, and skip flag.
             the input to the model are original and interpolated videos (t frames each of size n x n, processed with
             optical flow).
             skip flag indicates if a file has less frames than the number used for training.
    """

    # extracting data
    video_path = batch
    video_path = video_path[0]
    video_labels = torch.tensor([[0.0], [1.0]])  # original, sped-up
    n = random.randrange(64, 336)  # should be between 64 and 336 but it depends on gpu memory limit...
    interpolated_path = "/nas/home/smariani/video_interpolation/datasets/kinetics400/minterpolate_60fps/train/" \
                        + os.path.basename(video_path)
    # optical flow (will also keep just t frames and resize them to n x n)
    original_of = of(video_path, n, t)
    interp_of = of(interpolated_path, n, t)
    # building input tensor
    data, skip_file = generate_data(original_of, interp_of, n, t)
    return data, video_labels, skip_file


def val_data_processing(batch, n, t):
    """
    Data processing chain for test and validation videos.
    :param batch: batch containing original video path
    :param n: spatial dimension (frame size)
    :param t: temporal dimension (frame number)
    :return: input to the model(data, with batch size = 2), video label and skip flag.
             the input to the model are original and interpolated videos (t frames each of size n x n, processed with
             optical flow).
             skip flag indicates if a file has less frames than the number used for testing/validating.
    """

    # extracting data
    video_path = batch
    video_path = video_path[0]
    video_labels = torch.tensor([[0.0], [1.0]])  # original, sped-up
    interpolated_path = "/nas/home/smariani/video_interpolation/datasets/kinetics400/minterpolate_60fps/validation/" \
                        + os.path.basename(video_path)
    # optical flow (will also keep just t frames and resize them to n x n)
    original_of = of(video_path, n, t)
    interp_of = of(interpolated_path, n, t)
    # building input tensor
    data, skip_file = generate_data(original_of, interp_of, n, t)
    return data, video_labels, skip_file


def test_data_processing(batch, n, t):
    """
    Data processing chain for test  videos.
    :param batch: batch containing original video path
    :param n: spatial dimension (frame size)
    :param t: temporal dimension (frame number)
    :return: input to the model(data, with batch size = 2), video label and skip flag.
             the input to the model are original and interpolated videos (t frames each of size n x n, processed with
             optical flow).
             skip flag indicates if a file has less frames than the number used for testing/validating.
    """

    # extracting data
    video_path = batch
    video_path = video_path[0]
    video_labels = torch.tensor([[0.0], [1.0]])  # original, sped-up
    interpolated_path = "/nas/home/smariani/video_interpolation/datasets/kinetics400/minterpolate_60fps/test/" \
                        + os.path.basename(video_path)
    # optical flow (will also keep just t frames and resize them to n x n)
    original_of = of(video_path, n, t)
    interp_of = of(interpolated_path, n, t)
    # building input tensor
    data, skip_file = generate_data(original_of, interp_of, n, t)
    return data, video_labels, skip_file


def generate_data(f_list_1, f_list_2, n, t):
    """
    Generate tensor as input for the model.
    Starts from the frames lists of normal and 2x speed videos.
    Produces tensor of size [BATCH_SIZE(2), N_CHANNELS(3), N_FRAMES(t), HEIGHT(n), WIDTH(n)]
    :param f_list_1: frame list of original video with optical flow
    :param f_list_2: frame list of interpolated video with optical flow
    :param t: number of consecutive taken frames (temporal dimension)
    :param n: size of the frames (spatial dimension)
    :return: input tensor to feed the model and skip flag (if video has too low number of frames and has to be skipped)
    """
    skip_file = False
    data = None
    if len(f_list_1) < t or len(f_list_2) < t:
        skip_file = True
    else:
        frames_list = np.array([f_list_1, f_list_2])  # original, sped-up
        data = torch.autograd.Variable(torch.tensor(frames_list))
        data = torch.reshape(data, (2, t, n, n, 3))
        data = torch.permute(data, [0, 4, 1, 2, 3])
        data = data.float()

    return data, skip_file
