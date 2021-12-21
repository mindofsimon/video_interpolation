"""
Save video informations in csv file.
Set the speed manipulation parameter of the videos, along with the roots in which original videos are stored.

Each line of the csv contains:
    video path, video label (original = 0, manipulated = 1), video smp (if video is original is set to 1).
    train/test/validation videos will contain just original videos (sped-up versions will be created during training).
"""

import csv
import os
import cv2


# FOR SPEEDNET (KINETICS VIDEOS)
smp = 2
total_files = 10000  # then 90% will be used for training, 10% both for test and validation

# TRAIN
skipped_tra = 0
originals_root = '/nas/home/pbestagini/kinetics/k400/train/'
video_list = [v for v in os.listdir(originals_root) if v.endswith('.mp4')]
video_list = video_list[0:int(total_files*0.9)]
with open('/nas/home/smariani/video_interpolation/datasets/kinetics400/train.csv', mode='w', newline="") as videos:
    videos_writer = csv.writer(videos, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for v in video_list:
        if v.endswith('.mp4'):
            cap = cv2.VideoCapture(originals_root + v)
            if cap.isOpened():
                videos_writer.writerow([originals_root + v, "0", "1"])
            else:  # in case the file is corrupted, we skip it
                skipped_tra += 1

# TEST
skipped_tes = 0
originals_root = '/nas/home/pbestagini/kinetics/k400/test/'
video_list = [v for v in os.listdir(originals_root) if v.endswith('.mp4')]
video_list = video_list[0:int(total_files*0.1)]
with open('/nas/home/smariani/video_interpolation/datasets/kinetics400/test.csv', mode='w', newline="") as videos:
    videos_writer = csv.writer(videos, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for v in video_list:
        if v.endswith('.mp4'):
            cap = cv2.VideoCapture(originals_root + v)
            if cap.isOpened():
                videos_writer.writerow([originals_root + v, "0", "1"])
            else:  # in case the file is corrupted, we skip it
                skipped_tes += 1


# VALIDATION
skipped_val = 0
originals_root = '/nas/home/pbestagini/kinetics/k400/val/'
video_list = [v for v in os.listdir(originals_root) if v.endswith('.mp4')]
video_list = video_list[0:int(total_files*0.1)]
with open('/nas/home/smariani/video_interpolation/datasets/kinetics400/validation.csv', mode='w', newline="") as videos:
    videos_writer = csv.writer(videos, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for v in video_list:
        if v.endswith('.mp4'):
            cap = cv2.VideoCapture(originals_root + v)
            if cap.isOpened():
                videos_writer.writerow([originals_root + v, "0", "1"])
            else:  # in case the file is corrupted, we skip it
                skipped_val += 1

print("Skipped "+str(skipped_tra)+" files for training")
print("Skipped "+str(skipped_tes)+" files for testing")
print("Skipped "+str(skipped_val)+" files for validation")
