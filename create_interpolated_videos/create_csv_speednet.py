"""
Save video informations in csv file.
Each line of the csv contains the path to the video file.
All of the videos are original videos, the interpolated ones will be loaded at runtime during training/testing/validation.
Rememmber to set the root directories for the original files.
"""

import csv
import os


# FOR SPEEDNET (KINETICS VIDEOS)

# TRAIN
originals_root = '/nas/home/smariani/video_interpolation/datasets/kinetics400/originals/train/'
video_list = [v for v in os.listdir(originals_root)]
tot_tra = len(video_list)
with open('/nas/home/smariani/video_interpolation/datasets/kinetics400/train.csv', mode='w', newline="") as videos:
    videos_writer = csv.writer(videos, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for v in video_list:
        videos_writer.writerow([originals_root + v])
print("Total Train Original Videos : " + str(tot_tra))


# TEST
originals_root = '/nas/home/smariani/video_interpolation/datasets/kinetics400/originals/test/'
video_list = [v for v in os.listdir(originals_root)]
tot_tes = len(video_list)
with open('/nas/home/smariani/video_interpolation/datasets/kinetics400/test.csv', mode='w', newline="") as videos:
    videos_writer = csv.writer(videos, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for v in video_list:
        videos_writer.writerow([originals_root + v])
print("Total Test Original Videos : " + str(tot_tes))


# VALIDATION
originals_root = '/nas/home/smariani/video_interpolation/datasets/kinetics400/originals/validation/'
video_list = [v for v in os.listdir(originals_root)]
tot_val = len(video_list)
with open('/nas/home/smariani/video_interpolation/datasets/kinetics400/validation.csv', mode='w', newline="") as videos:
    videos_writer = csv.writer(videos, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for v in video_list:
        videos_writer.writerow([originals_root + v])
print("Total Validation Original Videos : " + str(tot_val))
