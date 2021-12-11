"""
If you have already trained SpeedNet, you can test it here without going through training again.
"""

from speednet import *
from torch_utils import *


# Input Parameters
T = 16  # frame number

# GPU parameters
set_gpu(0)
set_backend()
set_seed()
platf = platform()

# Model Parameters
SAVE_PATH = 'speednet.pth'
model = S3DG(num_classes=1, num_frames=T)
model.load_state_dict(torch.load(SAVE_PATH))
model.eval()
_, test_dl, _ = load_data()
testing(model, test_dl, platf, T)
