"""
Save video informations in csv file.
Set the speed manipulation parameter of the videos, along with the roots in which original and manipulated videos are stored.

Each line of the csv contains:
    video path, video label (original = 0, manipulated = 1), video smp (if video is original is set to 1).
"""

import csv
import os


# FOR SPEEDNET (KINETICS VIDEOS)
smp = 2

# TEST
originals_root = '/nas/home/smariani/video_interpolation/datasets/kinetics400/kinetics_videos/test/originals/'
manipulated_root = '/nas/home/smariani/video_interpolation/datasets/kinetics400/kinetics_videos/test/2x/'
with open('/nas/home/smariani/video_interpolation/datasets/kinetics400/kinetics_videos/test/test.csv', mode='w', newline="") as videos:
    videos_writer = csv.writer(videos, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for v in os.listdir(originals_root):
        videos_writer.writerow([originals_root + v, "0", "1"])
        videos_writer.writerow([manipulated_root + v, "1", smp])

# TRAIN
originals_root = '/nas/home/smariani/video_interpolation/datasets/kinetics400/kinetics_videos/train/originals/'
manipulated_root = '/nas/home/smariani/video_interpolation/datasets/kinetics400/kinetics_videos/train/2x/'
with open('/nas/home/smariani/video_interpolation/datasets/kinetics400/kinetics_videos/train/train.csv', mode='w', newline="") as videos:
    videos_writer = csv.writer(videos, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for v in os.listdir(originals_root):
        videos_writer.writerow([originals_root + v, "0", "1"])
        videos_writer.writerow([manipulated_root + v, "1", smp])

# VALIDATION
originals_root = '/nas/home/smariani/video_interpolation/datasets/kinetics400/kinetics_videos/validation/originals/'
manipulated_root = '/nas/home/smariani/video_interpolation/datasets/kinetics400/kinetics_videos/validation/2x/'
with open('/nas/home/smariani/video_interpolation/datasets/kinetics400/kinetics_videos/validation/validation.csv', mode='w', newline="") as videos:
    videos_writer = csv.writer(videos, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for v in os.listdir(originals_root):
        videos_writer.writerow([originals_root + v, "0", "1"])
        videos_writer.writerow([manipulated_root + v, "1", smp])
