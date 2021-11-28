"""
Building a video dataset from csv file.

For validation and testing datasets (training=False) we add all the files in the csv.
So validation and testing csv files will contain original and manipulated videos.

For training dataset (training=True) we add all the files in the csv (which consists in just the original files) then we will add manually the manipulated copy.
This to ensure that the batch extraction will be in order, so that each batch will contain (original, manipulated) version of a video
"""

from torch.utils.data import Dataset
import csv
import os


class VideoDataset(Dataset):
    def __init__(self, data_folder, manipulated_folder, training):
        self.samples = []

        with open(data_folder, 'r') as videos:
            data = csv.reader(videos, delimiter=',')
            for row in data:
                self.samples.append((row[0], row[1], row[2]))
                if training:
                    self.samples.append((str(manipulated_folder + os.path.basename(row[0])), str(1), str(2)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
