"""
Creating interpolated videos using FFMPEG filter 'minterpolate' at different manipulation parameters.
Since minterpolate with motion compensation is computationally expensive, a thread pool executor is used.
When the interpolation parameter is lower than the original video fps some frames are dropped.
In this filter, the video duration is the same for both input and output videos.
"""

from tqdm import tqdm
import ffmpeg
import os.path
from functools import partial
from concurrent.futures import ThreadPoolExecutor


def generate_video(filename, interp,  input_root, output_root):
    """
    Function to generate the video.
    :param filename: name of the input video file
    :param interp: interpolation parameter (output fps)
    :param input_root: input file directory (where to get it)
    :param output_root: output file directory (where to save it)
    :return: nothing
    """
    output_path = output_root + filename
    input_path = input_root + filename
    (
        ffmpeg
            .input(input_path)
            .filter('minterpolate', fps=str(interp), mi_mode='mci')
            .output(output_path)
            .global_args('-loglevel', 'quiet')  # just not to print ffmpeg header
            .run()
    )


def main():
    operations = ['test', 'validation', 'train']

    # Interpolation Factor
    i_fact = 60

    # Run ffmpeg on each video
    for op in operations:
        video_root = '/nas/home/smariani/video_interpolation/datasets/kinetics400/originals_temp/'
        output_video_root = '/nas/home/smariani/video_interpolation/datasets/kinetics400/minterpolate_60fps/'
        video_root = video_root + op + "/"
        output_video_root = output_video_root + op + "/"
        # Retrieve video list
        video_path_list = [v for v in os.listdir(video_root)]
        print("INTERPOLATION FACTOR: " + str(i_fact))
        video_root = video_root
        generate_video_partial = partial(generate_video, interp=i_fact, input_root=video_root, output_root=output_video_root)
        with ThreadPoolExecutor(16) as p:
            results_mthread = list(tqdm(p.map(generate_video_partial, video_path_list), total=len(video_path_list), desc='FFMPEG [minterpolate] multi-thread'))


if __name__ == '__main__':
    main()
