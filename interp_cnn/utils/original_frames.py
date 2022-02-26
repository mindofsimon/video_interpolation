import cv2
import albumentations as alb


def get_original_frames(video, n, t, training):
    """
    Extract first t original frames from a given video.
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

    frames_seq = []  # residuals sequence
    cap = cv2.VideoCapture(video)
    success, first_frame = cap.read()
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    h, w = first_frame.shape
    if training:
        frame_proc = alb.Compose([alb.Resize(height=n, width=n)])
        # if h*n < 224 or w*n < 224:
        #   frame_proc = alb.Compose([alb.Resize(height=224, width=224)])
        # else:
        #   frame_proc = alb.Compose([alb.Resize(height=int(n*h), width=int(n*w)), alb.CenterCrop(height=224, width=224)])
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
        # if h < 224 or w < 224:
        #   frame_proc = alb.Compose([alb.Resize(height=224, width=224)])
        # else:
        #   frame_proc = alb.Compose([alb.CenterCrop(height=224, width=224)])
    processed_frame = frame_proc(image=first_frame)
    frame = processed_frame["image"]
    frames_seq.append(frame / 255)
    success, actual_frame = cap.read()
    actual_frame = cv2.cvtColor(actual_frame, cv2.COLOR_BGR2GRAY)
    i = 1
    while success and i < t:
        # processing frame
        processed_frame = frame_proc(image=actual_frame)
        frame = processed_frame["image"]
        frames_seq.append(frame/255)
        success, actual_frame = cap.read()
        if success:
            actual_frame = cv2.cvtColor(actual_frame, cv2.COLOR_BGR2GRAY)
        i += 1

    return frames_seq
