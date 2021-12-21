"""
Save video informations in csv file.
Set the speed manipulation parameter of the videos, along with the roots in which original videos are stored.

Each line of the csv contains:
    video path, video label (original = 0, manipulated = 1), video smp (if video is original is set to 1).
    train/test/validation videos will contain just original videos (sped-up versions will be created during training).

For now we use 10000 videos for training and 1000 both for test and validation.
"""

import csv
import os


# FOR SPEEDNET (KINETICS VIDEOS)
smp = 2

# TEST
originals_root = '/nas/home/pbestagini/kinetics/k400/test/'
video_list = [v for v in os.listdir(originals_root) if v.endswith('.mp4')]
video_list = video_list[0:1000]
with open('/nas/home/smariani/video_interpolation/datasets/kinetics400/test.csv', mode='w', newline="") as videos:
    videos_writer = csv.writer(videos, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for v in video_list:
        if v.endswith('.mp4'):
            videos_writer.writerow([originals_root + v, "0", "1"])

# TRAIN
originals_root = '/nas/home/pbestagini/kinetics/k400/train/'
video_list = [v for v in os.listdir(originals_root) if v.endswith('.mp4')]
video_list = video_list[0:10000]
with open('/nas/home/smariani/video_interpolation/datasets/kinetics400/train.csv', mode='w', newline="") as videos:
    videos_writer = csv.writer(videos, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for v in video_list:
        if v.endswith('.mp4'):
            videos_writer.writerow([originals_root + v, "0", "1"])

# VALIDATION
originals_root = '/nas/home/pbestagini/kinetics/k400/val/'
video_list = [v for v in os.listdir(originals_root) if v.endswith('.mp4')]
video_list = video_list[0:1000]
with open('/nas/home/smariani/video_interpolation/datasets/kinetics400/validation.csv', mode='w', newline="") as videos:
    videos_writer = csv.writer(videos, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for v in video_list:
        if v.endswith('.mp4'):
            videos_writer.writerow([originals_root + v, "0", "1"])
