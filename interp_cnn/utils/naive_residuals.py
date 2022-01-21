from interp_cnn_utils import *


def get_naive_residuals(video, n, t):
    """
    Generate naive residuals sequence from a given video.
    Naive because it's just a subtraction between next and previous frames.
    Only t+1 video frames are taken in account (in order to have t residual frames),
    they are also resized to a n x n spatial dimension.
    :param video: video file path
    :param n: resizing each frame to n x n spatial dimension
    :param t: number of frames to extract
    :return: naive residual sequence
    """

    res_seq = []  # residuals sequence
    cap = cv2.VideoCapture(video)
    success, prev_frame = cap.read()
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    success, actual_frame = cap.read()
    actual_frame = cv2.cvtColor(actual_frame, cv2.COLOR_BGR2GRAY)
    i = 1
    while success and i < t+1:
        prev_frame = np.array(prev_frame)
        actual_frame = np.array(actual_frame)
        res = np.abs(np.subtract(actual_frame, prev_frame))
        # resizing image to n x n
        res = cv2.resize(res, dsize=(n, n), interpolation=cv2.INTER_NEAREST)
        res_seq.append(res/255)
        prev_frame = actual_frame
        success, actual_frame = cap.read()
        actual_frame = cv2.cvtColor(actual_frame, cv2.COLOR_BGR2GRAY)
        i += 1

    return res_seq
