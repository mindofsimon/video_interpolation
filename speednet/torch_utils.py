"""
Functions to handle GPU usage in PyTorch.
"""

import os
import numpy as np
import torch
import GPUtil
from torch.backends import cudnn


def set_gpu(id=-1):
    """
    Set tensor computation device.

    :param id: CPU or GPU device id (None for CPU, -1 for the device with lowest memory usage, or the ID)
    hint: use gpustat (pip install gpustat) in a bash CLI, or gputil (pip install gputil) in python.
    """
    if id is None:
        # CPU only
        print('GPU not selected')
        os.environ["CUDA_VISIBLE_DEVICES"] = str(-1)
    else:
        # -1 for automatic choice
        device = id if id != -1 else GPUtil.getFirstAvailable(order='memory')[0]
        try:
            name = GPUtil.getGPUs()[device].name
        except IndexError:
            print('The selected GPU does not exist. Switching to the most available one.')
            device = GPUtil.getFirstAvailable(order='memory')[0]
            name = GPUtil.getGPUs()[device].name
        print('GPU selected: %d - %s' % (device, name))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)


def torch_on_cuda():
    return os.environ["CUDA_VISIBLE_DEVICES"] and torch.cuda.is_available()


def set_backend():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def platform():
    if torch_on_cuda():
        # watch out! cuda for torch is 0 because it is the first torch can see! It is not the os.environ one!
        device = "cuda:0"
    else:
        device = "cpu"
    return torch.device(device)
