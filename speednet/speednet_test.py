"""
If you have already trained SpeedNet, you can test it here without going through training again.
"""

from speednet import *


# Input Parameters
N = 224  # frame size (N x N)
T = 16  # frame number

# Accuracy Parameters
true_positives = 0
total = 0
test_labels = []
predictions_list = []

# Model Parameters
SAVE_PATH = 'speednet.pth'
model = S3DG(num_classes=1, num_frames=T)
model.load_state_dict(torch.load(SAVE_PATH))
model.eval()
_, test_dl, _ = load_data()
testing(model, test_dl)

