"""
Utils for SpeedNet.
Including preprocessing steps for training/test/validation, data loading and output metrics.
"""

import cv2
from torch.utils.data import DataLoader
from load_data import VideoDataset
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np


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
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix (SPEEDNET)")
    plt.savefig('cm_speednet.png')
    plt.show()
    accuracy = round(((true_positives / total) * 100), 3)
    print("Accuracy:" + str(accuracy) + "%")


def load_data():
    """
    Loading data from csv files (path, label, speed manipulation parameter) to VideoDataset class.
    Then from VideoDataset class into DataLoaders.

    The train videos are loaded in double batch (original, manipulated).
    Test and validation videos are loaded in single batch.

    Be careful to put in originals csv the path to the video info csv files!

    :return: three data loaders (one for train videos, one for test videos and one for validation videos).
    """

    # train ds
    originals_csv = '/nas/home/smariani/video_interpolation/datasets/kinetics400/kinetics_videos/train/train.csv'
    dataset = VideoDataset(originals_csv)
    train_data_loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=2)

    # test ds
    originals_csv = '/nas/home/smariani/video_interpolation/datasets/kinetics400/kinetics_videos/test/test.csv'
    test_dataset = VideoDataset(originals_csv)
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=2)

    # valid ds
    originals_csv = '/nas/home/smariani/video_interpolation/datasets/kinetics400/kinetics_videos/validation/validation.csv'
    valid_dataset = VideoDataset(originals_csv)
    valid_data_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True, num_workers=2)

    return train_data_loader, test_data_loader, valid_data_loader


def center_crop(image):
    """
    Center crop (224 x 224) to apply to test and validation videos.
    :param image: input frame
    :return: cropped frame
    """

    center = []
    center.append(image.shape[0] / 2)
    center.append(image.shape[1] / 2)
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

    dim = None
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


def preprocess_train_video(vid, state, T, N):
    """
    Preprocessing operations applied to train videos.
    Spatial and temporal augmentations are applied.
    :param vid: input video
    :param state: 0 if video is original, 1 if video is manipulated
    :param T: temporal dimension (number of consecutive frames to take)
    :param N: spatial dimension (224)
    :return: list of T consecutive preprocessed frames
    """

    f_list = []
    cap = cv2.VideoCapture(vid)
    success, frame = cap.read()
    while success:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, dsize=(N, N), interpolation=cv2.INTER_NEAREST)  # spatial augmentation
        f_list.append(frame_rgb/255)
        success, frame = cap.read()

    if len(f_list) > (3*T):  # a video could have less than 3*T frames! (and also than T frames!)
        f_list = f_list[0:(3 * T)]

    if state == 1.0:  # sped up video
        f_list = f_list[::2]  # temporal augmentation (2 is for 2x speed videos!)
        # for the original videos we do nothing (like ::1)

    f_list = f_list[0:T]  # taking only T frames
    return f_list


def preprocess_test_video(vid, N, T):
    """
    Preprocessing operations applied to test and validation videos.
    No spatial and temporal augmentations.
    Image is resized to 224 pixels on the height maintaining ratio.
    Center cropping 224x224 is then applied.
    :param vid: input video
    :param N: spatial dimension (224)
    :param T: temporal dimension (number of consecutive frames to take)
    :return: list of T consecutive preprocessed frames
    """

    f_list = []
    cap = cv2.VideoCapture(vid)
    success, frame = cap.read()
    while success:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res_frame_rgb = image_resize(frame_rgb, height=N)
        if res_frame_rgb.shape[1] < N:  # in case width is lower than 224 pixels, we resize it to 224
            res_frame_rgb = cv2.resize(res_frame_rgb, dsize=(N, N), interpolation=cv2.INTER_NEAREST)
        crop_frame_rgb = center_crop(res_frame_rgb)
        f_list.append(crop_frame_rgb/255)
        success, frame = cap.read()

    f_list = f_list[0:T]  # taking only T frames
    return f_list


def test_val_data_processing(batch, N, T):
    """
    Data processing chain for test and validation videos.
    :param batch: batch containing video path, video label and video smp
    :param N: spatial dimension (frame size)
    :param T: temporal dimension (frame number)
    :return: input to the model(data, with batch size = 1), video label and skip flag.
             skip flag indicates if a file has less frames than the number used for training.
    """
    data = None
    skip_file = False
    video_path, video_label, _ = batch
    video_path = video_path[0]
    video_label = float(video_label[0])
    frames_list = preprocess_test_video(video_path, N, T)
    if len(frames_list) < T:
        skip_file = True
    else:
        frames_list = np.array([frames_list])
        data = torch.autograd.Variable(torch.tensor(frames_list))
        data = torch.reshape(data, (1, T, N, N, 3))
        data = torch.permute(data, [0, 4, 1, 2, 3])
        data = data.float()
    return data, video_label, skip_file


def train_data_processing(batch, N, T):
    """
    Data processing chain for train videos.
    :param batch: batch containing video path, video label and video smp
    :param N: spatial dimension (frame size)
    :param T: temporal dimension (frame number)
    :return: input to the model(data, with batch size = 2), video labels, and skip flag.
             skip flag indicates if a file has less frames than the number used for training.
    """

    data = None
    skip_file = False
    video_path, video_label, _ = batch
    video_path_1 = video_path[0]
    video_path_2 = video_path[1]
    video_label_1 = float(video_label[0])
    video_label_2 = float(video_label[1])
    video_labels = torch.tensor([[video_label_1], [video_label_2]])
    # building input tensor
    frames_list_1 = preprocess_train_video(video_path_1, video_label_1, T, N)
    frames_list_2 = preprocess_train_video(video_path_2, video_label_2, T, N)
    if len(frames_list_1) < T or len(frames_list_2) < T:
        skip_file = True
    else:
        frames_list = np.array([frames_list_1, frames_list_2])
        data = torch.autograd.Variable(torch.tensor(frames_list))
        data = torch.reshape(data, (2, T, N, N, 3))
        data = torch.permute(data, [0, 4, 1, 2, 3])
        data = data.float()
    return data, video_labels, skip_file
