"""
Save video informations in csv file.
Each line of the csv contains the path to the video file.
All of the videos are original videos, manipulated (sped up) videos will be created at runtime during training)
Rememmber to set the root directories for the original files.
"""

import csv
import os
import cv2


# FOR SPEEDNET (KINETICS VIDEOS)
total_files = 30000  # then 90% will be used for training, 10% both for test and validation

# TRAIN
skipped_tra = 0
ok_tra = 0
originals_root = '/nas/home/pbestagini/kinetics/k400/train/'
video_list = [v for v in os.listdir(originals_root) if v.endswith('.mp4')]
video_list = video_list[0:int(total_files*0.9)]
tot_tra = len(video_list)
with open('/nas/home/smariani/video_interpolation/datasets/kinetics400/train.csv', mode='w', newline="") as videos:
    videos_writer = csv.writer(videos, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for v in video_list:
        if v.endswith('.mp4'):
            cap = cv2.VideoCapture(originals_root + v)
            if cap.isOpened():
                videos_writer.writerow([originals_root + v])
                ok_tra += 1
            else:  # in case the file is corrupted, we skip it
                skipped_tra += 1

# TEST
skipped_tes = 0
ok_tes = 0
originals_root = '/nas/home/pbestagini/kinetics/k400/test/'
video_list = [v for v in os.listdir(originals_root) if v.endswith('.mp4')]
video_list = video_list[0:int(total_files*0.1)]
tot_tes = len(video_list)
with open('/nas/home/smariani/video_interpolation/datasets/kinetics400/test.csv', mode='w', newline="") as videos:
    videos_writer = csv.writer(videos, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for v in video_list:
        if v.endswith('.mp4'):
            cap = cv2.VideoCapture(originals_root + v)
            if cap.isOpened():
                videos_writer.writerow([originals_root + v])
                ok_tes += 1
            else:  # in case the file is corrupted, we skip it
                skipped_tes += 1


# VALIDATION
skipped_val = 0
ok_val = 0
originals_root = '/nas/home/pbestagini/kinetics/k400/val/'
video_list = [v for v in os.listdir(originals_root) if v.endswith('.mp4')]
video_list = video_list[0:int(total_files*0.1)]
tot_val = len(video_list)
with open('/nas/home/smariani/video_interpolation/datasets/kinetics400/validation.csv', mode='w', newline="") as videos:
    videos_writer = csv.writer(videos, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for v in video_list:
        if v.endswith('.mp4'):
            cap = cv2.VideoCapture(originals_root + v)
            if cap.isOpened():
                videos_writer.writerow([originals_root + v])
                ok_val += 1
            else:  # in case the file is corrupted, we skip it
                skipped_val += 1

print("TRAIN\nTotal videos: "+str(tot_tra)+" Valid videos: "+str(ok_tra)+" Corrupted videos: "+str(skipped_tra))
print("TEST\nTotal videos: "+str(tot_tes)+" Valid videos: "+str(ok_tes)+" Corrupted videos: "+str(skipped_tes))
print("VALIDATION\nTotal videos: "+str(tot_val)+" Valid videos: "+str(ok_val)+" Corrupted videos: "+str(skipped_val))
