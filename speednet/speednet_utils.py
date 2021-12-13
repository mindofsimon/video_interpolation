"""
Utils for SpeedNet.
Including preprocessing steps for training/test/validation, data loading and output metrics.
"""

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

    All videos are loaded in single batch (containing: video path, video label, video smp).
    Train/Test/Validation videos dataloaders will contain just original videos (2x speed version will be created during training phase)

    Be careful to put in originals csv the path to the video info csv files!

    :return: three data loaders (one for train videos, one for test videos and one for validation videos).
    """

    # train ds
    videos_csv = '/nas/home/smariani/video_interpolation/datasets/kinetics400/kinetics_videos/train/train.csv'
    dataset = VideoDataset(videos_csv)
    train_data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    # test ds
    videos_csv = '/nas/home/smariani/video_interpolation/datasets/kinetics400/kinetics_videos/test/test.csv'
    test_dataset = VideoDataset(videos_csv)
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=2)

    # valid ds
    videos_csv = '/nas/home/smariani/video_interpolation/datasets/kinetics400/kinetics_videos/validation/validation.csv'
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


def preprocess_train_video(vid, n, t):
    """
    Preprocessing operations applied to train videos.
    Also the 2x speed version of the original video is created by subsampling video frames.
    Spatial and temporal augmentations are applied.
    :param vid: input video
    :param t: temporal dimension (number of consecutive frames to take)
    :param n: spatial dimension (random number in range (64, 336))
    :return: two lists of t consecutive preprocessed frames (one for normal speed, one for 2x speed)
    """

    f_list = []  # containing original video frames
    cap = cv2.VideoCapture(vid)
    success, frame = cap.read()
    while success:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # spatial augmentation (resizing image to n x n)
        frame_rgb = cv2.resize(frame_rgb, dsize=(n, n), interpolation=cv2.INTER_NEAREST)
        f_list.append(frame_rgb/255)
        success, frame = cap.read()

    if len(f_list) > (3*t):  # a video could have less than 3*T frames! (and also than T frames!)
        f_list = f_list[0:(3 * t)]

    # temporal augmentation (sampling frames at different skip probabilities to create normal and sped-up video)
    factor_1 = round(random.uniform(1.0, 1.2), 2)
    prob_1 = 1 - (1 / factor_1)
    factor_2 = round(random.uniform(1.7, 2.2), 2)
    prob_2 = 1 - (1 / factor_2)
    random_numbers = np.random.uniform(0, 1, len(f_list))
    valid_frames_1 = np.where(random_numbers > prob_1)
    valid_frames_2 = np.where(random_numbers > prob_2)

    f_list_1 = np.array(f_list)
    f_list_2 = np.array(f_list)
    f_list_1 = f_list_1[valid_frames_1]  # normal speed
    f_list_2 = f_list_2[valid_frames_2]  # 2x speed

    # taking t frames only
    f_list_1 = list(f_list_1)
    f_list_2 = list(f_list_2)
    f_list_1 = f_list_1[0:t]
    f_list_2 = f_list_2[0:t]
    return f_list_1, f_list_2


def preprocess_test_val_video(vid, n, t):
    """
    Preprocessing operations applied to test and validation videos.
    Also the 2x speed version of the original video is created by subsampling video frames.
    No spatial and temporal augmentations.
    Image is resized to 224 pixels on the height maintaining ratio.
    Center cropping 224x224 is then applied.
    :param vid: input video
    :param n: spatial dimension (224)
    :param t: temporal dimension (number of consecutive frames to take)
    :return: two lists of t consecutive preprocessed frames (one for normal speed, one for 2x speed)
    """

    f_list_1 = []
    cap = cv2.VideoCapture(vid)
    success, frame = cap.read()
    while success:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res_frame_rgb = image_resize(frame_rgb, height=n)
        if res_frame_rgb.shape[1] < n:  # in case width is lower than 224 pixels, we resize it to 224
            res_frame_rgb = cv2.resize(res_frame_rgb, dsize=(n, n), interpolation=cv2.INTER_NEAREST)
        crop_frame_rgb = center_crop(res_frame_rgb)
        f_list_1.append(crop_frame_rgb/255)
        success, frame = cap.read()

    f_list_2 = f_list_1[::2]
    f_list_2 = f_list_2[0:t]  # taking only T frames
    f_list_1 = f_list_1[0:t]  # taking only T frames
    return f_list_1, f_list_2


def train_data_processing(batch, t):
    """
    Data processing chain for train videos.
    From video path to input (to the model) tensor.
    :param batch: batch containing video path, video label and video smp
    :param t: temporal dimension (frame number)
    :return: input to the model(data, with batch size = 2), video labels, and skip flag.
             skip flag indicates if a file has less frames than the number used for training.
    """

    video_path, _, _ = batch
    video_path = video_path[0]
    video_labels = torch.tensor([[0.0], [1.0]])  # original, sped-up
    # building input tensor
    n = random.randrange(64, 336)  # should be between 64 and 336 but it depends on gpu memory limit...
    frames_list_1, frames_list_2 = preprocess_train_video(video_path, n, t)
    data, skip_file = generate_data(frames_list_1, frames_list_2, n, t)
    return data, video_labels, skip_file


def test_val_data_processing(batch, n, t):
    """
    Data processing chain for test and validation videos.
    :param batch: batch containing video path, video label and video smp
    :param n: spatial dimension (frame size)
    :param t: temporal dimension (frame number)
    :return: input to the model(data, with batch size = 2), video label and skip flag.
             skip flag indicates if a file has less frames than the number used for testing/validating.
    """

    video_path, _, _ = batch
    video_path = video_path[0]
    video_labels = torch.tensor([[0.0], [1.0]])  # original, sped-up
    frames_list_1, frames_list_2 = preprocess_test_val_video(video_path, n, t)
    data, skip_file = generate_data(frames_list_1, frames_list_2, n, t)
    return data, video_labels, skip_file


def generate_data(f_list_1, f_list_2, n, t):
    """
    Generate tensor as input for the model.
    Starts from the frames lists of normal and 2x speed videos.
    Produces tensor of size [BATCH_SIZE(2), N_CHANNELS(3), N_FRAMES(t), HEIGHT(n), WIDTH(n)]
    :param f_list_1: frame list of original video
    :param f_list_2: frame list of sped up video
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
