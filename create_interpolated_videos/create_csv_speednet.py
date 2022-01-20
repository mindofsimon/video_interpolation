"""
Save video informations in csv file.
Each line of the csv contains the path to the video file.
All of the videos are original videos, the interpolated ones will be loaded at runtime during training/testing/validation.
A video is written just if it has enough frames (MIN_FRAMES).
Rememmber to set the root directories for the original files.
"""

import csv
import os
import cv2
from tqdm import tqdm


MIN_FRAMES = 32


def enough_frames(video_path):
    """
    Controls if a video has a minimum number of frames.
    :param video_path: video file path
    :return: True if enough frames, False otherwise
    """
    frame_list = []
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    i = 0
    while success and i < MIN_FRAMES:
        frame_list.append(frame)
        success, frame = cap.read()
        i += 1
    if len(frame_list) == MIN_FRAMES:
        valid = True
    else:
        valid = False
    return valid


# TRAIN
valid_videos = 0
originals_root = '/nas/home/smariani/video_interpolation/datasets/kinetics400/originals/train/'
video_list = [v for v in os.listdir(originals_root)]
tot_tra = len(video_list)
with open('/nas/home/smariani/video_interpolation/datasets/kinetics400/train.csv', mode='w', newline="") as videos:
    videos_writer = csv.writer(videos, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for v in tqdm(video_list, total=tot_tra, desc='writing training videos'):
        if enough_frames(originals_root + v):
            videos_writer.writerow([originals_root + v])
            valid_videos += 1
print("Total Train Original Videos : " + str(tot_tra))
print("Total Train Original VALID Videos : " + str(valid_videos))


# TEST
valid_videos = 0
originals_root = '/nas/home/smariani/video_interpolation/datasets/kinetics400/originals/test/'
video_list = [v for v in os.listdir(originals_root)]
tot_tes = len(video_list)
with open('/nas/home/smariani/video_interpolation/datasets/kinetics400/test.csv', mode='w', newline="") as videos:
    videos_writer = csv.writer(videos, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for v in tqdm(video_list, total=tot_tes, desc='writing test videos'):
        if enough_frames(originals_root + v):
            videos_writer.writerow([originals_root + v])
            valid_videos += 1
print("Total Test Original Videos : " + str(tot_tes))
print("Total Test Original VALID Videos : " + str(valid_videos))


# VALIDATION
valid_videos = 0
originals_root = '/nas/home/smariani/video_interpolation/datasets/kinetics400/originals/validation/'
video_list = [v for v in os.listdir(originals_root)]
tot_val = len(video_list)
with open('/nas/home/smariani/video_interpolation/datasets/kinetics400/validation.csv', mode='w', newline="") as videos:
    videos_writer = csv.writer(videos, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for v in tqdm(video_list, total=tot_val, desc='writing validation videos'):
        if enough_frames(originals_root + v):
            videos_writer.writerow([originals_root + v])
            valid_videos += 1
print("Total Validation Original Videos : " + str(tot_val))
print("Total Validation Original VALID Videos : " + str(valid_videos))
