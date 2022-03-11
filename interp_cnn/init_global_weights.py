"""
Initializing wieghts of Global Speednet with weights of frames/residuals/opticalflow models.
"""
import torch

from networks.global_speednet_torch import S3DG as S3DG_GLOB
from networks.speednet_torch import S3DG as S3DG_SING
from utils.torch_utils import *


# Input Parameters
N_CHANNELS = 1  # number of channels per frame (with optical flow is 3, with residuals is 1)
T = 16  # frames number
N = 224  # frames size (N x N)


def global_state_dict(platf):
    save_path_sing_1 = '/nas/home/smariani/video_interpolation/interp_cnn/models/speednet_frames_2.pth'
    save_path_sing_2 = '/nas/home/smariani/video_interpolation/interp_cnn/models/speednet_residuals_2.pth'
    save_path_sing_3 = '/nas/home/smariani/video_interpolation/interp_cnn/models/speednet_opticalflow_2.pth'
    model_glob = S3DG_GLOB(num_classes=1, num_frames=T, input_channels=N_CHANNELS)
    model_sing_1 = S3DG_SING(num_classes=1, num_frames=T, input_channels=N_CHANNELS)
    model_sing_2 = S3DG_SING(num_classes=1, num_frames=T, input_channels=N_CHANNELS)
    model_sing_3 = S3DG_SING(num_classes=1, num_frames=T, input_channels=N_CHANNELS)
    model_sing_1.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(save_path_sing_1).items()})
    model_sing_2.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(save_path_sing_2).items()})
    model_sing_3.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(save_path_sing_3).items()})
    model_glob = model_glob.to(platf)
    model_sing_1 = model_sing_1.to(platf)
    model_sing_2 = model_sing_2.to(platf)
    model_sing_3 = model_sing_3.to(platf)
    glob_state = model_glob.state_dict()
    sing_state_1 = model_sing_1.state_dict()
    sing_state_2 = model_sing_2.state_dict()
    sing_state_3 = model_sing_3.state_dict()
    params_1 = []
    params_2 = []
    params_3 = []
    for key in sing_state_1.keys():
        parameter = sing_state_1[key]
        params_1.append(parameter)
    for key in sing_state_2.keys():
        parameter = sing_state_2[key]
        params_2.append(parameter)
    for key in sing_state_3.keys():
        parameter = sing_state_3[key]
        params_3.append(parameter)
    params = params_1[0:len(params_1)-2] + params_2[0:len(params_2)-2] + params_3[0:len(params_3)-2] +\
             params_1[len(params_1)-2:] + params_2[len(params_2)-2:] + params_3[len(params_3)-2:]
    params.append(glob_state['final_linear.weight'])
    params.append(glob_state['final_linear.bias'])
    # params.append(torch.tensor([0.342, 0.546, 0.414]))
    # params.append(torch.tensor([0.0]))
    i = 0
    for key in glob_state.keys():
        glob_state[key] = params[i]
        i += 1
    return glob_state
