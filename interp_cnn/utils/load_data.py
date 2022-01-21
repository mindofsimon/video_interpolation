"""
Building a video dataset from csv file.
csv file must contain 3 columns per row (path, class, speed manipulation parameter).
"""

from torch.utils.data import Dataset
import csv


class VideoDataset(Dataset):
    """
    Video Dataset class
    """
    def __init__(self, csv_path):
        """
        Class constructor
        :param csv_path: path to the csv file containing video information
        """
        self.samples = []

        with open(csv_path, 'r') as videos:
            data = csv.reader(videos, delimiter=',')
            for row in data:
                self.samples.append((row[0]))

    def __len__(self):
        """
        :return: dataset length
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        :param idx: position of an element in the dataset
        :return: element in the dataset in position idx
        """
        return self.samples[idx]
