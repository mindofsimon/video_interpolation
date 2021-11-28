"""
Creating interpolated videos using FFMPEG filter 'setpts'.
If smp > 1 --> sped up video (dropping frames)
If smp < 1 --> slowed down video
This filter keeps the same frame rate between input and output videos, it just adds or removes frames altering the video duration (and so speed).
"""

import ffmpeg
import os.path


def generate_video(path, par, out_root):
    """
    Function to generate the video.
    :param path: path to the video to manipulate
    :param par: speed manipulation parameter
    :param out_root: output file path (where to save it)
    :return: nothing
    """
    filename = os.path.basename(path)
    factor = round((1/par), 3)
    (
        ffmpeg
            .input(path)
            .filter('setpts', str(factor)+'*PTS')
            .output(out_root+filename)
            .run()
    )


smp = 2  # speed manipulation parameter (2 means doubling speed)
root = '/'  # video root folder
output_root = root+"output/"  # video output root
for video in os.listdir(root):
    generate_video(root+video, smp, output_root)
