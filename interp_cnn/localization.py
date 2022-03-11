"""
Trying to scan T frames by N x N pixels blocks and predict if a block is interpolated or not.
"""
import torch.nn as nn
from utils.torch_utils import *
from networks.speednet_torch import S3DG
from interp_cnn_utils import *
import cv2
import math
from tqdm import tqdm


# Input Parameters
N_CHANNELS = 1  # number of channels per frame (with optical flow is 3, with residuals is 1)
T = 16  # frames number
N = 224  # frames size (N x N)
MULTI_GPU_TRAINING = True  # if model was trained with multi GPU
MULTI_GPU_TESTING = False  # if we want to test with double GPU
BATCH_SIZE = 8


def highlight_interpolated_blocks(frames, model, plat):
    residuals_seq = get_residuals(frames)
    residuals_arr = np.array(residuals_seq)
    h, w = residuals_arr[0].shape
    mask = np.zeros([h, w])
    for i in range(math.floor(h / N)):
        for j in range(math.floor(w / N)):
            block = residuals_arr[:, i * N:(i + 1) * N, j * N:(j + 1) * N]
            pred = test_block(block, model, plat)
            mask[i * N:(i + 1) * N, j * N:(j + 1) * N] = 1 - int(pred)
        # coviering also remaining column (rest of w/N...)
        block = residuals_arr[:, i * N:(i + 1) * N, w - N:]
        pred = test_block(block, model, plat)
        mask[i * N:(i + 1) * N, w - N:] = 1 - int(pred)
    # covering also remaining row (rest of h/N...)
    for j in range(math.floor(w / N)):
        block = residuals_arr[:, h - N:, j * N:(j + 1) * N]
        pred = test_block(block, model, plat)
        mask[h - N:, j * N:(j + 1) * N] = 1 - int(pred)
    block = residuals_arr[:, h - N: , w - N:]
    pred = test_block(block, model, plat)
    mask[h - N:, w - N:] = 1 - int(pred)
    highlight = frames.copy()
    for f in highlight:
        f[:, :, 0] = mask * f[:, :, 0]
        f[:, :, 1] = mask * f[:, :, 1]
    return highlight

def local_interpolation(original, interpolated):
    original_frames = extract_frames(original)
    interpolated_frames = extract_frames(interpolated)
    original_frames = np.array(original_frames)
    interpolated_frames = np.array(interpolated_frames)
    original_frames[:, 0:224, 0:224] = interpolated_frames[0:len(original_frames), 0:224, 0:224]
    return original_frames


def extract_frames(video):
    frames_seq = []  # residuals sequence
    cap = cv2.VideoCapture(video)
    success, actual_frame = cap.read()
    # i = 0
    # while i < T + 1 and success:
    while success:
        frames_seq.append(actual_frame)
        success, actual_frame = cap.read()
        # i += 1
    return frames_seq


def get_residuals(frames):
    frames_gray = []
    for f in frames:
        frames_gray.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    frames_1 = np.array(frames_gray[1:])
    frames_2 = np.array(frames_gray[0:-1])
    residuals = np.subtract(frames_1, frames_2)
    residuals = residuals/255
    return residuals


def test_block(prep_block, model, plat):
    preprocessed_block = np.array(prep_block)
    block_tensor = torch.autograd.Variable(torch.Tensor(preprocessed_block))
    block_tensor = torch.reshape(block_tensor, (1, T, N, N, N_CHANNELS))
    block_tensor = torch.permute(block_tensor, [0, 4, 1, 2, 3])
    block_tensor = block_tensor.float()
    block_tensor = block_tensor.to(plat)
    logits = model(block_tensor)
    video_class = torch.round(torch.sigmoid(logits)).item()
    return video_class  # 0.0: original, 1.0: interpolated


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
    _, test_dl, _ = load_data(BATCH_SIZE)
    test_interp_dir = '/nas/home/smariani/video_interpolation/datasets/kinetics400/minterpolate/test/'
    total_predictions = 0
    real_labels = []
    predicted_labels = []
    for batch in tqdm(test_dl, total=len(test_dl), desc='localization test'):
        for v in batch:
            original_video = v
            interpolated_video = test_interp_dir + os.path.basename(v)
            locally_int_video = local_interpolation(original_video, interpolated_video)
            residuals_seq = get_residuals(locally_int_video)
            residuals_arr = np.array(residuals_seq)
            h, w = residuals_arr[0].shape
            if h >= N and w >= N:
                first_block = True
                for i in range(math.floor(h/N)):
                    for j in range(math.floor(w/N)):
                        block = residuals_arr[:, i*N:(i+1)*N, j*N:(j+1)*N]
                        pred = test_block(block, model, plat)
                        predicted_labels.append(pred)
                        total_predictions += 1
                        if first_block:
                            first_block = False
                            real_labels.append(1.0)
                        else:
                            real_labels.append(0.0)
    correct_predictions = np.sum(np.array(predicted_labels) == np.array(real_labels))
    print_eval_metrics(real_labels, predicted_labels, correct_predictions, total_predictions)


if __name__ == '__main__':
    # main()
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
    model = model.to(plat)
    model.eval()
    vid = './car.mp4'
    video_frames = extract_frames(vid)
    # GYidQlv_IRM_000022_0000320.mp4
    # orig = '/nas/home/smariani/video_interpolation/datasets/kinetics400/originals/test/GYidQlv_IRM_000022_0000320.mp4'
    # interp = '/nas/home/smariani/video_interpolation/datasets/kinetics400/minterpolate/test/GYidQlv_IRM_000022_0000320.mp4'
    # video_frames = local_interpolation(orig, interp)
    highlighted_frames = []
    out = cv2.VideoWriter('./loc_out.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (video_frames[0].shape[1], video_frames[0].shape[0]))
    for i in range(math.floor(len(video_frames)/(T+1))):  # need T+1 frames to create T residuals!
        highlighted_frames.append(highlight_interpolated_blocks(video_frames[i*(T+1):(i+1)*(T+1)], model, plat))
        for j in range(len(highlighted_frames[i])):
            out.write(np.array(highlighted_frames[i][j]))
    out.release()
