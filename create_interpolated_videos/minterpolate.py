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


def generate_video(path, interp, output_root):
    """
    Function to generate the video.
    :param path: path to the video to interpolate
    :param interp: interpolation parameter (output fps)
    :param output_root: output file path (where to save it)
    :return: nothing
    """
    filename = os.path.basename(path)
    output_folder = output_root + str(interp) + 'fps_minterpolate/' + filename
    (
        ffmpeg
            .input(path)
            .filter('minterpolate', fps=str(interp), mi_mode='mci')
            .output(output_folder)
            .global_args('-loglevel', 'quiet')  # just not to print ffmpeg header
            .run()
    )


def main():
    # Folder parameters (put paths to original and manipulated train/test videos
    # be careful to set the right paths
    root = ''
    out_root = root+'/output/'
    video_root = root+'/videos/'
    output_video_root = out_root+'/out_videos/'

    # Ffmpeg params
    ffmpeg_interp_list = [15, 45, 60]

    # Retrieve video list
    video_path_list = []
    for path in os.listdir(video_root):
        video_path_list.append(video_root+path)

    # Run ffmpeg on each video
    for interp in ffmpeg_interp_list:
        print("INTERPOLATION FACTOR: " + str(interp))
        generate_video_partial = partial(generate_video, interp=interp, output_root=output_video_root)
        with ThreadPoolExecutor(2) as p:
            results_mthread = list(tqdm(p.map(generate_video_partial, video_path_list), total=len(video_path_list), desc='Multi-threading'))


if __name__ == '__main__':
    main()
