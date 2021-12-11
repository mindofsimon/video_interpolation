"""
Save video informations in csv file.
Set the speed manipulation parameter of the videos, along with the roots in which original videos are stored.

Each line of the csv contains:
    video path, video label (original = 0, manipulated = 1), video smp (if video is original is set to 1).
    train/test/validation videos will contain just original videos (sped-up versions will be created during training).
"""

import csv
import os


# FOR SPEEDNET (KINETICS VIDEOS)
smp = 2

# TEST
originals_root = '/nas/home/smariani/video_interpolation/datasets/kinetics400/kinetics_videos/test/originals/'
with open('/nas/home/smariani/video_interpolation/datasets/kinetics400/kinetics_videos/test/test.csv', mode='w', newline="") as videos:
    videos_writer = csv.writer(videos, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for v in os.listdir(originals_root):
        videos_writer.writerow([originals_root + v, "0", "1"])

# TRAIN
originals_root = '/nas/home/smariani/video_interpolation/datasets/kinetics400/kinetics_videos/train/originals/'
with open('/nas/home/smariani/video_interpolation/datasets/kinetics400/kinetics_videos/train/train.csv', mode='w', newline="") as videos:
    videos_writer = csv.writer(videos, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for v in os.listdir(originals_root):
        videos_writer.writerow([originals_root + v, "0", "1"])

# VALIDATION
originals_root = '/nas/home/smariani/video_interpolation/datasets/kinetics400/kinetics_videos/validation/originals/'
with open('/nas/home/smariani/video_interpolation/datasets/kinetics400/kinetics_videos/validation/validation.csv', mode='w', newline="") as videos:
    videos_writer = csv.writer(videos, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for v in os.listdir(originals_root):
        videos_writer.writerow([originals_root + v, "0", "1"])
