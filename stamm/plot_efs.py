"""
Plotting EFS (Encoded Frame Size Sequence) of a given video.
Insert in the variable 'path' the path to the video you want to display.
You need common_functions.py to have all the processing functions to make it work.
The plot is composed by colored dots (Green = I frame, Red = B frame, Orange = P frame)
indicating the weight in bytes of each frame of the video.
"""

import matplotlib.patches as mpatches
from common_functions import *


def plot_efs(frame_size_list, color_list):  # plots a video encoded frame size sequence
    plt.figure(figsize=(20, 10), dpi=300)

    # PLOTS INTERPOLATING LINE
    plt.plot(list(range(1, len(frame_size_list)+1)), frame_size_list, color='blue', linewidth="0.5")

    # PLOTS POINTS (MARKERS) WITH DIFFERENT COLOR DEPENDING ON THE FRAME TYPE
    plt.scatter(list(range(1, len(frame_size_list)+1)), frame_size_list, c=color_list, marker='o')
    plt.title('Encoded Frame Sequence')
    plt.xlabel('frames')
    plt.ylabel('bytes')

    # ADDING PATCHES TO IDENTIFY FRAME TYPE
    green_patch = mpatches.Patch(color='green', label='I frames')
    red_patch = mpatches.Patch(color='red', label='B frames')
    orange_patch = mpatches.Patch(color='orange', label='P frames')
    plt.legend(handles=[green_patch, red_patch, orange_patch])
    # plt.savefig('efs_seq_' + os.path.basename(path) + '.png')
    plt.show()


path = '/nas/home/smariani/video_interpolation/datasets/kinetics400/kinetics_videos/test/originals/-ZRoSFQGAiA_000056_000066.mp4'  # insert here the path of the video to display
f_sizes, _, f_colors = extract_frames_info(path)
plot_efs(f_sizes, f_colors)

