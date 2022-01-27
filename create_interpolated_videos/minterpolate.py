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

    # Run ffmpeg on each video
    for op in operations:
        video_root = '/nas/home/smariani/video_interpolation/datasets/kinetics400/originals_temp/'
        output_video_root = '/nas/home/smariani/video_interpolation/datasets/kinetics400/minterpolate/'
        video_root = video_root + op + "/"
        output_video_root = output_video_root + op + "/"
        # Retrieve video list
        video_path_list = [v for v in os.listdir(video_root)]
        video_path_list = sorted(video_path_list)
        total_elements = len(video_path_list)
        first_part = video_path_list[0:int(total_elements/3)]
        second_part = video_path_list[int(total_elements/3):int(2*total_elements/3)]
        third_part = video_path_list[int(2*total_elements/3):]
        generate_video_partial_1 = partial(generate_video, interp=60, input_root=video_root, output_root=output_video_root)
        with ThreadPoolExecutor(16) as p_1:
            results_mthread = list(tqdm(p_1.map(generate_video_partial_1, first_part), total=len(first_part), desc='FFMPEG [minterpolate 60fps] multi-thread ' + op.upper()))
        generate_video_partial_2 = partial(generate_video, interp=50, input_root=video_root, output_root=output_video_root)
        with ThreadPoolExecutor(16) as p_2:
            results_mthread = list(tqdm(p_2.map(generate_video_partial_2, second_part), total=len(second_part), desc='FFMPEG [minterpolate 50fps] multi-thread ' + op.upper()))
        generate_video_partial_3 = partial(generate_video, interp=40, input_root=video_root, output_root=output_video_root)
        with ThreadPoolExecutor(16) as p_3:
            results_mthread = list(tqdm(p_3.map(generate_video_partial_3, third_part), total=len(third_part), desc='FFMPEG [minterpolate 40fps] multi-thread ' + op.upper()))


if __name__ == '__main__':
    main()
