from interp_cnn_utils import *
import cv2
import albumentations as alb


def get_naive_residuals(video, n, t, training):
    """
    Generate naive residuals sequence from a given video.
    Naive because it's just a subtraction between next and previous frames.
    Only t+1 video frames are taken in account (in order to have t residual frames).
    Train video frames and Test/Validation video frames are processed in different ways.
    Train:
        - each frame is resized to n x n
    Test/Validation:
        - each frame is resized to a height of n and a width to keep the same ratio as the original frame
            - if the obtained width is less than n we resize the frame to n x n
            - if the obtained width is greater than n we apply a n x n center crop
    :param video: video file path
    :param n: resizing each frame to n x n spatial dimension
    :param t: number of frames to extract
    :param training: if function is called during training (will apply different processing to frames based on this)
    :return: naive residual sequence
    """

    res_seq = []  # residuals sequence
    cap = cv2.VideoCapture(video)
    success, prev_frame = cap.read()
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    h, w = prev_frame.shape
    if training:
        frame_proc = alb.Compose([alb.Resize(height=n, width=n)])
    else:
        # in test and validation we apply a resizement 224 x (224*w/h) and then a 224 x 224 center cropping
        # if the resizement produces a width < 224, we just resize everything to 224 x 224,
        # without applying center cropping
        ratio = h / w
        if round(n / ratio) < n:
            # resizing to 224 x 224 if resizement would produce a width < 224
            frame_proc = alb.Compose([alb.Resize(height=n, width=n)])  # just resizing to nxn if video is too small
        else:
            # resizing keeping same ratio (height set to 224) and center cropping (224 x 224)
            # if resized width is ok (>224), center cropping
            frame_proc = alb.Compose([alb.Resize(height=n, width=round(n / ratio)), alb.CenterCrop(height=n, width=n)])
    success, actual_frame = cap.read()
    actual_frame = cv2.cvtColor(actual_frame, cv2.COLOR_BGR2GRAY)
    i = 1
    while success and i < t+1:
        prev_frame = np.array(prev_frame)
        actual_frame = np.array(actual_frame)
        res = np.abs(np.subtract(actual_frame, prev_frame))
        # processing frame
        processed_frame = frame_proc(image=res)
        res = processed_frame["image"]
        res_seq.append(res/255)
        prev_frame = actual_frame
        success, actual_frame = cap.read()
        actual_frame = cv2.cvtColor(actual_frame, cv2.COLOR_BGR2GRAY)
        i += 1

    return res_seq
