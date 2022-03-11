"""
If you have already trained a model, you can test videos with this script.
"""
from interp_cnn import *


# Input Parameters
N_CHANNELS = 1  # number of channels per frame (with optical flow is 3, with residuals is 1)
T = 16  # frames number
N = 224  # frames size (N x N)
MULTI_GPU_TRAINING = True  # if model was traied with multi GPU
MULTI_GPU_TESTING = False  # if we want to test with double GPU


def test_video(video, model, plat):
    """
    Test interpolation cnn on a single file
    :param video: video file path
    :param model: trained model
    :param plat: platform (GPU)
    :return: 0 if video is original, 1 if video is interpolated
    """
    preprocessed_video = get_naive_residuals(video, N, T, training=False)
    preprocessed_video = np.array(preprocessed_video)
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
        set_gpu(1)
    set_backend()
    set_seed()
    plat = platform()

    # Model Parameters
    save_path = '/nas/home/smariani/video_interpolation/interp_cnn/models/speednet.pth'
    model = S3DG(num_classes=1, num_frames=T, input_channels=N_CHANNELS)
    if MULTI_GPU_TRAINING:
        model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(save_path).items()})
    else:
        model.load_state_dict(torch.load(save_path))
    if MULTI_GPU_TESTING:
        model = nn.DataParallel(model)  # (multi GPU)
    model = model.to(plat)
    model.eval()
    vid = '/nas/home/pbestagini/video_interpolation_detection/test_orig.mp4'  # video to test
    pred = test_video(vid, model, plat)
    print(pred)


if __name__ == '__main__':
    main()
