"""
Extracts train/test/validation video files from kinetics400 dataset tar.gz archives.
"""
import tarfile
import os


partitions = 10
for p in range(partitions):
    fname = '/nas/home/pbestagini/kinetics/k400/train/part_'+str(p)+'.tar.gz'
    if fname.endswith("tar.gz"):
        tar = tarfile.open(fname, "r:gz")
        tar.extractall('/nas/home/smariani/video_interpolation/datasets/kinetics_400/train/')
        tar.close()
    elif fname.endswith("tar"):
        tar = tarfile.open(fname, "r:")
        tar.extractall('/nas/home/smariani/video_interpolation/datasets/kinetics_400/train/')
        tar.close()
    os.remove('/nas/home/smariani/video_interpolation/datasets/kinetics_400/train/part_' + str(p) + '.tar')

for p in range(partitions):
    fname = '/nas/home/pbestagini/kinetics/k400/test/part_'+str(p)+'.tar.gz'
    if fname.endswith("tar.gz"):
        tar = tarfile.open(fname, "r:gz")
        tar.extractall('/nas/home/smariani/video_interpolation/datasets/kinetics_400/test/')
        tar.close()
    elif fname.endswith("tar"):
        tar = tarfile.open(fname, "r:")
        tar.extractall('/nas/home/smariani/video_interpolation/datasets/kinetics_400/test/')
        tar.close()
    os.remove('/nas/home/smariani/video_interpolation/datasets/kinetics_400/test/part_' + str(p) + '.tar')

for p in range(partitions):
    fname = '/nas/home/pbestagini/kinetics/k400/validation/part_' + str(p) + '.tar.gz'
    if fname.endswith("tar.gz"):
        tar = tarfile.open(fname, "r:gz")
        tar.extractall('/nas/home/smariani/video_interpolation/datasets/kinetics_400/validation/')
        tar.close()
    elif fname.endswith("tar"):
        tar = tarfile.open(fname, "r:")
        tar.extractall('/nas/home/smariani/video_interpolation/datasets/kinetics_400/validation/')
        tar.close()
    os.remove('/nas/home/smariani/video_interpolation/datasets/kinetics_400/validation/part_' + str(p) + '.tar')
