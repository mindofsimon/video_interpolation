"""
Dense Optical Flow.
"""
import cv2 as cv
import numpy as np


def of(vid, n, t):
    """
    Applies optical flow to first t video frames.
    Frames are also resized to n x n.
    :param vid: video filename
    :param n: spatial dimension
    :param t: temporal dimension
    :return: optical flow frames
    """

    flow_frames = []

    # The video feed is read in as
    # a VideoCapture object
    cap = cv.VideoCapture(vid)

    # ret = a boolean return value from
    # getting the frame, first_frame = the
    # first frame in the entire video sequence
    ret, first_frame = cap.read()

    # Converts frame to grayscale because we
    # only need the luminance channel for
    # detecting edges - less computationally
    # expensive
    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

    # Creates an image filled with zero
    # intensities with the same dimensions
    # as the frame
    mask = np.zeros_like(first_frame)

    # Sets image saturation to maximum
    mask[..., 1] = 255

    # Extracting second frame
    ret, frame = cap.read()
    i = 1
    while ret and i < t:

        # Converts each frame to grayscale - we previously
        # only converted the first frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Calculates dense optical flow by Farneback method
        flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Computes the magnitude and angle of the 2D vectors
        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])

        # Sets image hue according to the optical flow
        # direction
        mask[..., 0] = angle * 180 / np.pi / 2

        # Sets image value according to the optical flow
        # magnitude (normalized)
        mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

        # Converts HSV to RGB (BGR) color representation
        rgb_frame_of = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
        # spatial augmentation (resizing image to n x n)
        rgb_frame_of = cv.resize(rgb_frame_of, dsize=(n, n), interpolation=cv.INTER_NEAREST)
        flow_frames.append(rgb_frame_of/255)

        # Updates previous frame
        prev_gray = gray

        # Read new frame
        ret, frame = cap.read()
        i += 1

    cap.release()
    return flow_frames
