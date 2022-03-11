"""
If you have already trained an interpolation cnn, you can test it here without going through training again.
"""

from interp_cnn import *
from utils.torch_utils import *
#from global_interp_cnn_utils import *
from networks.speednet_torch import S3DG as RES
from networks.global_speednet_torch import S3DG as GLOB
#from tqdm import tqdm
#import torch.nn as nn
from init_global_weights import global_state_dict


# Input Parameters
BATCH_SIZE_LOAD = 2  # how many original videos are extracted from the dataloaders at a time
BATCH_SIZE_MODEL = 2 * BATCH_SIZE_LOAD  # how many elements are fed into the model at a time (original + manipulated)
N_CHANNELS = 1  # number of channels per frame (with optical flow is 3, with residuals is 1)
T = 16  # frames number
N = 224  # frames size (N x N)
MULTI_GPU_TRAINING = True  # if model was traied with multi GPU


def testing_2(model, test_dl, platf, t):
    # TESTING
    # Accuracy Parameters
    true_positives = 0
    total = 0
    all_video_classes = []
    all_video_labels = []

    with torch.no_grad():
        for batch in tqdm(test_dl, total=len(test_dl), desc='testing'):
            data, video_labels = test_data_processing(batch, N, t, N_CHANNELS)
            # moving data to platform
            data = data.to(platf)
            video_labels = video_labels.to(platf)
            # predicting logits
            logits = model(data)
            total += BATCH_SIZE_MODEL
            med_logits = torch.mean(logits, 1)
            video_classes = torch.round(torch.sigmoid(med_logits))
            video_classes = video_classes.tolist()
            # flat_list_video_classes = [item for sublist in video_classes for item in sublist]
            # all_video_classes = all_video_classes + flat_list_video_classes
            all_video_classes = all_video_classes + video_classes
            video_classes = np.array(video_classes)
            video_labels = video_labels.tolist()
            flat_list_video_labels = [item for sublist in video_labels for item in sublist]
            all_video_labels = all_video_labels + flat_list_video_labels
            video_labels = np.array(flat_list_video_labels)
            true_positives += np.sum(video_classes == video_labels)
    # EVALUATION METRICS
    print_eval_metrics(all_video_labels, all_video_classes, true_positives, total)


# GPU parameters
# set_gpu(1)
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
set_backend()
set_seed()
platf = platform()

# Model Parameters
save_path_model = '/nas/home/smariani/video_interpolation/interp_cnn/models/speednet_global_2.pth'
model = S3DG(num_classes=1, num_frames=T, input_channels=N_CHANNELS)
#model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(save_path_model).items()})
#save_path_res = '/nas/home/smariani/video_interpolation/interp_cnn/models/speednet_residuals_2.pth'
#model_res = RES(num_classes=1, num_frames=T, input_channels=N_CHANNELS)
#model_res.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(save_path_res).items()})
# glob_dict = model.state_dict()
# glob_dict = global_state_dict(platf)
#res_dict = model_res.state_dict()
#for key in res_dict.keys():
#    print(key)
#print(glob_dict['residuals.18.weight'])
#print(res_dict['features.18.weight'])

if MULTI_GPU_TRAINING:
    #model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(save_path_model).items()})
    #model.state_dict(global_state_dict(platf))
    model.load_state_dict(global_state_dict(platf))
else:
    model.load_state_dict(torch.load(save_path_model))
model = nn.DataParallel(model)  # (multi GPU)
model = model.to(platf)
model.eval()
_, test_dl, _ = load_data(BATCH_SIZE_LOAD)
# testing(model, test_dl, platf, T)
testing_2(model, test_dl, platf, T)
