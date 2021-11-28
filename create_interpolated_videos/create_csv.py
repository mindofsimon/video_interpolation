"""
Save video informations in csv file.
Set the speed manipulation parameter of the videos, along with the roots in which original and manipulated videos are stored.

Each line of the csv contains:
    video path, video label (original = 0, manipulated = 1), video smp (if video is original is set to 1).
"""

import csv
import os


smp = 2
originals_root = '/'
manipulated_root = '/'
with open('videos.csv', mode='w', newline="") as videos:
    videos_writer = csv.writer(videos, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for v in os.listdir(originals_root):
        videos_writer.writerow([originals_root + v, "0", "1"])
    for v in os.listdir(manipulated_root):
        videos_writer.writerow([manipulated_root + v, "1", smp])
