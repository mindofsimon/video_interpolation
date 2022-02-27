"""
Complete SpeedNet implementation.
It includes Training, Validation and Testing.
The model is trained to discriminate between original videos and videos interpolated at 60/50/40fps (with minterpolate
filter from FFMPEG, consider that original videos are at 30fps).
The model takes as input a tensor of size [BATCH_SIZE_MODEL, N_CHANNELS, N_FRAMES (T), HEIGHT (N), WIDTH (N)].
Input (original) videos are loaded through data loaders (check load_data.py and interp_cnn_utils.py for more details).
Interpolated videos are loaded at runtime from their folder (they share the same filename as the originals
but they are placed in a different folder).
Then both videos are processed producing optical flow (or residuals) only on first T frames.
Resulting frames are then resized to a N x N dimension for training (n is random in training), for test/validation
instead, frames are resized to a height of N and a width in order to keep the same ratio, then a center crop is applied.
The data is then sent into the model.
Model is trained with BCE loss with logits.
You can choose to train on multi GPU and also to continue with a previous training cycle.
"""

import torch.nn as nn
import torch.optim as optim
# from networks.speednet_torch import S3DG
from networks.global_speednet_torch import S3DG
# from interp_cnn_utils import *
from global_interp_cnn_utils import *
import pandas as pd
from utils.torch_utils import *
from tqdm import tqdm


# Input Parameters
BATCH_SIZE_LOAD = 2  # how many original videos are extracted from the dataloaders at a time
BATCH_SIZE_MODEL = 2 * BATCH_SIZE_LOAD  # how many elements are fed into the model at a time (original + interpolated)
# so in this way we extract BATCH_SIZE_LOAD original videos from the dataloaders at a time.
# then we load the interpolated version of those extracted videos, so we got a total of 2*BATCH_SIZE_LOAD videos.
# finally we fed those videos into the model.
N_CHANNELS = 1  # number of channels per frame (with optical flow is 3, with residuals is 1)
T = 16  # frames number
N = 224  # frames size (N x N)
MULTI_GPU_TRAINING = True  # if we want to train with double GPU
NEW_TRAINING_CYCLE = True  # if a new training starts or instead if we continue from a previous checkpoint


def training(optimizer, criterion, model, train_dl, platf):
    """
    Training function
    :param optimizer: model optimizer
    :param criterion: model loss function
    :param model: actual model
    :param train_dl: train data loader
    :param platf : data will be loaded on this platform
    :return: nothing
    """

    for batch in tqdm(train_dl, total=len(train_dl), desc='training'):
        data, video_labels = train_data_processing(batch, T, N_CHANNELS)
        # moving data to platform
        data = data.to(platf)
        video_labels = video_labels.to(platf)
        # predicting logits
        logits = model(data)
        # calculating loss and optimizing
        loss = criterion(logits, video_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # logits<0 --> class 0 (so logit<0, label=0 low loss; logit<0, label=1 high loss)
        # logits>0 --> class 1 (so logit>0, label=0 high loss; logit>0, label=1 low loss)


def validating(criterion, model, valid_dl, platf):
    """
    Validation function
    :param criterion: loss function
    :param model: actual model
    :param valid_dl: validation data loader
    :param platf : data will be loaded on this platform
    :return: nothing
    """

    # VALIDATION
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in tqdm(valid_dl, total=len(valid_dl), desc='validating'):
            data, video_labels = val_data_processing(batch, N, T, N_CHANNELS)
            # moving data to platform
            data = data.to(platf)
            video_labels = video_labels.to(platf)
            # predicting logits
            logits = model(data)
            # calculating loss
            val_loss += criterion(logits, video_labels).item()
            video_classes = torch.round(torch.sigmoid(logits))
            video_classes = video_classes.tolist()
            video_classes = np.array(video_classes)
            video_labels = video_labels.tolist()
            video_labels = np.array(video_labels)
            correct += np.sum(video_classes == video_labels)
        val_loss /= (len(valid_dl))
        correct /= (len(valid_dl.dataset) * 2)
        print(f"validation error: \n accuracy: {(100 * correct):>0.1f}%, avg loss: {val_loss:>8f}")
    return correct, val_loss


def testing(model, test_dl, platf, t):
    """
    Test function
    :param model: actual model
    :param test_dl: test data loader
    :param platf : data will be loaded on this platform
    :param t : segment of frames to be tested for each video
    :return: nothing
    """

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
            video_classes = torch.round(torch.sigmoid(logits))
            video_classes = video_classes.tolist()
            flat_list_video_classes = [item for sublist in video_classes for item in sublist]
            all_video_classes = all_video_classes + flat_list_video_classes
            video_classes = np.array(video_classes)
            video_labels = video_labels.tolist()
            flat_list_video_labels = [item for sublist in video_labels for item in sublist]
            all_video_labels = all_video_labels + flat_list_video_labels
            video_labels = np.array(video_labels)
            true_positives += np.sum(video_classes == video_labels)
    # EVALUATION METRICS
    print_eval_metrics(all_video_labels, all_video_classes, true_positives, total)


def main():
    save_path = '/nas/home/smariani/video_interpolation/interp_cnn/models/speednet_global_2.pth'
    model = S3DG(num_classes=1, num_frames=T, input_channels=N_CHANNELS)
    # GPU parameters
    if MULTI_GPU_TRAINING:
        os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
    else:
        set_gpu(0)
    set_backend()
    set_seed()
    platf = platform()
    # Model Parameters
    if MULTI_GPU_TRAINING:
        model = nn.DataParallel(model)  # just to train faster (multi GPU)
        if not NEW_TRAINING_CYCLE:
            model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(save_path).items()})
    else:
        if not NEW_TRAINING_CYCLE:
            model.load_state_dict(torch.load(save_path))
    model.to(platf)
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-02)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, mode='min', verbose=True)
    epochs = 15
    best_acc = 0
    no_improvement = 0     # n of epochs with no improvements
    patience = 7          # max n of epoch with no improvements
    min_val_loss = np.inf
    history = []

    # TRAINING
    train_dl, test_dl, valid_dl = load_data(BATCH_SIZE_LOAD)
    print("TRAINING SPEEDNET")
    for e in range(epochs):
        print("EPOCH "+str(e+1))
        model.train()
        training(optimizer, criterion, model, train_dl, platf)
        # VALIDATING
        model.eval()
        correct, val_loss = validating(criterion, model, valid_dl, platf)
        lr_scheduler.step(val_loss)
        history.append({"epoch": e, "accuracy": 100*correct, "loss": val_loss, "lr": optimizer.param_groups[0]['lr']})
        # MODEL CHECKPOINT CALLBACK
        accuracy = correct
        if accuracy > best_acc:
            # Callback for weight saving
            torch.save(model.state_dict(), save_path)
            best_acc = accuracy

        # EARLY STOP CALLBACK
        if val_loss < min_val_loss:  # Improvement in the new epoch
            no_improvement = 0
            min_val_loss = val_loss
        else:  # No improvement in the new epoch
            no_improvement += 1

        if e > 5 and no_improvement == patience:  # Patience reached
            print(f'Early stopped at epoch {e}')
            # Save history for early stopping
            df = pd.DataFrame(history)
            df.to_csv("/nas/home/smariani/video_interpolation/interp_cnn/eval_metrics/speednet_global_2_history.csv")
            break

    print("Done!")
    # Save history
    df = pd.DataFrame(history)
    df.to_csv("/nas/home/smariani/video_interpolation/interp_cnn/eval_metrics/speednet_global_2_history.csv")

    # TESTING
    print("TESTING SPEEDNET")
    model.eval()
    testing(model, test_dl, platf, T)


if __name__ == '__main__':
    main()
