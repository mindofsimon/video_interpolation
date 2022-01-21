"""
You can use this function if you want to test the interpolation cnn on a single file.
"""
from interp_cnn import *
from utils.dense_optical_flow import of

# Input Parameters
BATCH_SIZE_LOAD = 1  # how many original videos are extracted from the dataloaders at a time
BATCH_SIZE_MODEL = 2 * BATCH_SIZE_LOAD  # how many elements are fed into the model at a time (original + manipulated)
N_CHANNELS = 3  # number of channels per frame (with optical flow is 3, with residuals is 1)
T = 32  # frames number
N = 224  # frames size (N x N)
NET_TYPE = "speednet"  # can be speednet or re_net


def test_video(video):
    """
    Test interpolation cnn on a single file
    :param video: video file path
    :return: 0 if video is original, 1 if video is interpolated
    """
    # GPU parameters
    set_gpu(0)
    set_backend()
    set_seed()
    plat = platform()

    # Model Parameters
    if NET_TYPE == 'speednet':
        save_path = '/nas/home/smariani/video_interpolation/interp_cnn/models/speednet.pth'
        model = S3DG(num_classes=1, num_frames=T, input_channel=N_CHANNELS)
    else:  # re_net
        save_path = '/nas/home/smariani/video_interpolation/interp_cnn/models/re_net.pth'
        model = ReNet(n_frames=T, spatial_dim=N, in_channels=N_CHANNELS)
    model.load_state_dict(torch.load(save_path))
    model.to(plat)
    model.eval()
    preprocessed_video = of(video, N, T)
    video_tensor = torch.Tensor(preprocessed_video)
    video_tensor.to(plat)
    logits = model(video_tensor)
    manip_prob = torch.round(torch.sigmoid(logits)).item()
    return int(manip_prob)
