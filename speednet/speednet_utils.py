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
    print(test_labels)
    print(predictions_list)
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
    Manipulated_fold is useless if training=False, see load_data.py for more details...

    Be careful to put in originals csv the path to the video info csv files!

    :return: three data loaders (one for train videos, one for test videos and one for validation videos).
    """

    # train ds
    originals_csv = 'C:/Users/maria/OneDrive/Desktop/polimi/MAE/thesis/create_interpolated_videos/speednet_train_videos.csv'
    manipulated_fold = 'C:/Users/maria/OneDrive/Desktop/polimi/MAE/thesis/kinetics400/kinetics_videos/train/2x/'
    dataset = VideoDataset(originals_csv, manipulated_fold, training=True)
    train_data_loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=2)

    # test ds
    originals_csv = 'C:/Users/maria/OneDrive/Desktop/polimi/MAE/thesis/create_interpolated_videos/speednet_test_videos.csv'
    test_dataset = VideoDataset(originals_csv, manipulated_fold, training=False)
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=2)

    # valid ds
    originals_csv = 'C:/Users/maria/OneDrive/Desktop/polimi/MAE/thesis/create_interpolated_videos/speednet_validation_videos.csv'
    valid_dataset = VideoDataset(originals_csv, manipulated_fold, training=False)
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

    f_list = f_list[0:(3 * T)]
    if state == 1.0:  # sped up video
        f_list = f_list[::2]  # temporal augmentation
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
