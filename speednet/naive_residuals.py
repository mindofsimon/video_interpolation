import cv2
import random
import numpy as np


def get_naive_residuals(video, n, t):
    """
    Generate naive residuals sequence from a given video.
    Naive because it's just a subtraction between next and previous frames.
    Only t video frames are taken in account, they are also resized to a n x n spatial dimension.
    :param video: video file path
    :param n: resizing each frame to n x n spatial dimension
    :param t: number of frames to extract
    :return: naive residual sequence
    """

    f_list = []  # containing original video frames
    cap = cv2.VideoCapture(video)
    success, frame = cap.read()
    i = 0
    while success and i < t:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        f_list.append(frame_rgb/255)
        success, frame = cap.read()
        i += 1

    # calculating naive residuals
    list_1 = np.array(f_list[1:])
    list_2 = np.array(f_list[0:-1])
    residuals_sequence = list(abs(list_1 - list_2))
    residuals_sequence_resized = []
    for r in residuals_sequence:
        # spatial augmentation (resizing image to n x n)
        residuals_sequence_resized.append(cv2.resize(r, dsize=(n, n), interpolation=cv2.INTER_NEAREST))
    return residuals_sequence_resized
