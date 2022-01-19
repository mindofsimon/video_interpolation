"""
Split a video in equal chunks (if possible).
The split is performed at the nearest keyframe, in a way to avoid re-encoding.
"""
from tqdm import tqdm
import ffmpeg
import os.path
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import cv2


def split_vid(filename, input_root, output_root):
    """
    Splits a video
    :param filename: name of the input video file
    :param input_root: input file directory (where to get it)
    :param output_root: output file directory (where to save it)
    :return: nothing
    """
    input_path = input_root + filename
    (
        ffmpeg
            .input(input_path)
            .output(output_root+os.path.splitext(filename)[0] + '%d.mp4', f='segment', segment_time='3', reset_timestamps='1', c='copy')
            .global_args('-loglevel', 'quiet')
            .run()
    )


def main():
    video_root = '/nas/home/pbestagini/kinetics/k400/train/'
    output_video_root = '/nas/home/smariani/video_interpolation/datasets/kinetics400/originals_temp/train/'

    # Retrieve video list
    total_files = 10000
    video_path_list = [v for v in os.listdir(video_root) if v.endswith('.mp4')]
    video_path_list = video_path_list[10000:10000 + int(total_files * 0.9)]
    for v in video_path_list:
        cap = cv2.VideoCapture(video_root + v)
        if not cap.isOpened():
            video_path_list.remove(v)
    split_vid_partial = partial(split_vid, input_root=video_root, output_root=output_video_root)
    with ThreadPoolExecutor(16) as p:
        results_mthread = list(tqdm(p.map(split_vid_partial, video_path_list), total=len(video_path_list), desc='FFMPEG [split] multi-thread'))


if __name__ == '__main__':
    main()
