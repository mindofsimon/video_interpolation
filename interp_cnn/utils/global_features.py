import cv2
import albumentations as alb
import numpy as np


def extract_of(frames):
    flow_frames = []
    prev_gray = frames[0]
    colored_frame = cv2.cvtColor(prev_gray, cv2.COLOR_GRAY2BGR)
    mask = np.zeros_like(colored_frame)
    mask[..., 1] = 255
    gray = frames[1]
    i = 1
    while i < len(frames):
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mask[..., 0] = angle * 180 / np.pi / 2
        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        rgb_frame_of = cv2.cvtColor(mask, cv2.COLOR_HSV2RGB)
        gray_frame_of = cv2.cvtColor(rgb_frame_of, cv2.COLOR_RGB2GRAY)
        flow_frames.append(gray_frame_of/255)
        prev_gray = gray
        if i + 1 < len(frames):
            gray = frames[i + 1]
        i += 1

    return flow_frames


def extract_residuals(frames):
    res_seq = []
    prev_frame = frames[0]
    actual_frame = frames[1]
    i = 1
    while i < len(frames):
        prev_frame = np.array(prev_frame)
        actual_frame = np.array(actual_frame)
        res = np.abs(np.subtract(actual_frame, prev_frame))
        res_seq.append(res/255)
        prev_frame = actual_frame
        if i + 1 < len(frames):
            actual_frame = frames[i + 1]
        i += 1
    return res_seq


def extract_frames(video, n, t, training):
    frames_seq = []  # residuals sequence
    cap = cv2.VideoCapture(video)
    success, first_frame = cap.read()
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    h, w = first_frame.shape
    if training:
        if h * n < 224 or w * n < 224:
            frame_proc = alb.Compose([alb.Resize(height=224, width=224)])
        else:
            frame_proc = alb.Compose(
                [alb.Resize(height=int(n * h), width=int(n * w)), alb.CenterCrop(height=224, width=224)])
    else:
        if h < 224 or w < 224:
            frame_proc = alb.Compose([alb.Resize(height=224, width=224)])
        else:
            frame_proc = alb.Compose([alb.CenterCrop(height=224, width=224)])
    processed_frame = frame_proc(image=first_frame)
    frame = processed_frame["image"]
    frames_seq.append(frame)
    success, actual_frame = cap.read()
    actual_frame = cv2.cvtColor(actual_frame, cv2.COLOR_BGR2GRAY)
    i = 1
    while success and i < t + 1:
        # processing frame
        processed_frame = frame_proc(image=actual_frame)
        frame = processed_frame["image"]
        frames_seq.append(frame)
        success, actual_frame = cap.read()
        if success:
            actual_frame = cv2.cvtColor(actual_frame, cv2.COLOR_BGR2GRAY)
        i += 1
    return frames_seq


def extract_global_features(video, n, t, training):
    frames = extract_frames(video, n, t, training)
    residuals = extract_residuals(frames)
    optflow = extract_of(frames)
    frames = np.array(frames)
    frames = frames[0:t]  # cause extraction is till t + 1...to create residuals/optflow sequences of t frames
    frames = frames / 255
    return frames, residuals, optflow
