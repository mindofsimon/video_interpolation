"""
If you have already trained an interpolation cnn, you can test it here without going through training again.
"""

from interp_cnn import *
from utils.torch_utils import *


# Input Parameters
BATCH_SIZE_LOAD = 8  # how many original videos are extracted from the dataloaders at a time
BATCH_SIZE_MODEL = 2 * BATCH_SIZE_LOAD  # how many elements are fed into the model at a time (original + manipulated)
N_CHANNELS = 1  # number of channels per frame (with optical flow is 3, with residuals is 1)
T = 16  # frames number
N = 224  # frames size (N x N)
NET_TYPE = "speednet"  # can be speednet or re_net (or also speednet_alt, an alternative version of S3DG)
MULTI_GPU_TRAINING = True  # if model was traied with multi GPU

# GPU parameters
set_gpu(1)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
set_backend()
set_seed()
platf = platform()

# Model Parameters
if NET_TYPE == 'speednet':
    save_path = '/nas/home/smariani/video_interpolation/interp_cnn/models/speednet.pth'
    model = S3DG(num_classes=1, num_frames=T, input_channels=N_CHANNELS)
elif NET_TYPE == 're_net':  # re_net
    save_path = '/nas/home/smariani/video_interpolation/interp_cnn/models/re_net.pth'
    model = ReNet(n_frames=T, spatial_dim=N, in_channels=N_CHANNELS)
else:  # speednet_alt
    save_path = '/nas/home/smariani/video_interpolation/interp_cnn/models/speednet.pth'
    model = S3DG_alt(num_classes=1, num_frames=T, input_channels=N_CHANNELS)
if MULTI_GPU_TRAINING:
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(save_path).items()})
else:
    model.load_state_dict(torch.load(save_path))
# model = nn.DataParallel(model)  # (multi GPU)
model.to(platf)
model.eval()
_, test_dl, _ = load_data(BATCH_SIZE_LOAD)
testing(model, test_dl, platf, T, NET_TYPE)
