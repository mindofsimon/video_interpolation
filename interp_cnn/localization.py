"""
Trying to scan 16 frames by 224 x 224 pixels blocks and predict if a block is interpolated or not.
"""
from interp_cnn import *
import cv2
import math


# Input Parameters
N_CHANNELS = 1  # number of channels per frame (with optical flow is 3, with residuals is 1)
T = 16  # frames number
N = 224  # frames size (N x N)
MULTI_GPU_TRAINING = True  # if model was traied with multi GPU
MULTI_GPU_TESTING = False  # if we want to test with double GPU


def get_residuals(video, start_frame):
    res_seq = []  # residuals sequence
    cap = cv2.VideoCapture(video)
    success, prev_frame = cap.read()
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    success, actual_frame = cap.read()
    actual_frame = cv2.cvtColor(actual_frame, cv2.COLOR_BGR2GRAY)
    while success:
        prev_frame = np.array(prev_frame)
        actual_frame = np.array(actual_frame)
        res = np.abs(np.subtract(actual_frame, prev_frame))
        res_seq.append(res / 255)
        prev_frame = actual_frame
        success, actual_frame = cap.read()
        if success:
            actual_frame = cv2.cvtColor(actual_frame, cv2.COLOR_BGR2GRAY)

    return res_seq[start_frame:start_frame + T]


def test_video(prep_video, model, plat):
    """
    Test interpolation cnn on a single file
    :param prep_video: video matrix
    :param model: trained model
    :param plat: platform (GPU)
    :return: 0 if video is original, 1 if video is interpolated
    """

    preprocessed_video = np.array(prep_video)
    video_tensor = torch.autograd.Variable(torch.Tensor(preprocessed_video))
    video_tensor = torch.reshape(video_tensor, (1, T, N, N, N_CHANNELS))
    video_tensor = torch.permute(video_tensor, [0, 4, 1, 2, 3])
    video_tensor = video_tensor.float()
    video_tensor = video_tensor.to(plat)
    logits = model(video_tensor)
    video_class = torch.round(torch.sigmoid(logits)).item()
    return int(video_class)  # 0: original, 1: interpolated


def main():
    # GPU parameters
    if MULTI_GPU_TESTING:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    else:
        set_gpu(0)
    set_backend()
    set_seed()
    plat = platform()

    # Model Parameters
    save_path = '/nas/home/smariani/video_interpolation/interp_cnn/models/speednet_residuals_2.pth'
    model = S3DG(num_classes=1, num_frames=T, input_channels=N_CHANNELS)
    if MULTI_GPU_TRAINING:
        model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(save_path).items()})
    else:
        model.load_state_dict(torch.load(save_path))
    if MULTI_GPU_TESTING:
        model = nn.DataParallel(model)  # (multi GPU)
    model.to(plat)
    model.eval()
    vid = '/nas/home/smariani/video_interpolation/interp_cnn/car.mp4'  # video to test
    residuals_seq = get_residuals(vid, start_frame=90)
    residuals_arr = np.array(residuals_seq)
    h, w = residuals_arr[0].shape
    for i in range(math.floor(h/N)):
        for j in range(math.floor(w/N)):
            res = residuals_arr[:, i*N:(i+1)*N, j*N:(j+1)*N]
            pred = test_video(res, model, plat)
            print(pred)


if __name__ == '__main__':
    main()
