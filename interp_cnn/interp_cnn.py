"""
Complete Interpolation CNN implementation.
Can choose among two models (SPEEDNET or RE_NET).
It includes Training, Validation and Testing.
The model takes as input a tensor of size [BATCH_SIZE_MODEL, N_CHANNELS, N_FRAMES (T), HEIGHT (N), WIDTH (N)].
Input (original) videos are loaded through data loaders (check load_data.py and interp_cnn_utils.py for more details).
Interpolated videos are loaded at runtime from their folder (they share the same filename as the originals
but they are placed in a different folder).
Then both videos are processed producing optical flow (or residuals) only on first T frames.
Resulting frames are then resized to a N x N dimension.
The data is then sent into the model (speednet or re_net).
Model is trained with BCE loss with logits.
"""

import torch.nn as nn
import torch.optim as optim
from networks.speednet_torch import S3DG
from networks.re_net import ReNet
from interp_cnn_utils import *
import pandas as pd
from utils.torch_utils import *
from tqdm import tqdm


# Input Parameters
BATCH_SIZE_LOAD = 1  # how many original videos are extracted from the dataloaders at a time
BATCH_SIZE_MODEL = 2 * BATCH_SIZE_LOAD  # how many elements are fed into the model at a time (original + manipulated)
N_CHANNELS = 3  # number of channels per frame (with optical flow is 3, with residuals is 1)
T = 32  # frames number
N = 224  # frames size (N x N)
NET_TYPE = "speednet"  # can be speednet or re_net


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
        data, video_labels = train_data_processing(batch, N, T, N_CHANNELS)
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
            manipulation_probs = torch.round(torch.sigmoid(logits))
            video_labels = video_labels.tolist()
            manipulation_probs = np.array(manipulation_probs)
            video_labels = np.array(video_labels)
            correct = np.sum(manipulation_probs == video_labels)
        val_loss /= (len(valid_dl))
        correct /= (len(valid_dl.dataset) * 2)
        print(f"validation error: \n accuracy: {(100 * correct):>0.1f}%, avg loss: {val_loss:>8f}")
    return correct, val_loss


def testing(model, test_dl, platf, t, net):
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
    test_labels = []
    predictions_list = []

    with torch.no_grad():
        for batch in tqdm(test_dl, total=len(test_dl), desc='testing'):
            data, video_labels = test_data_processing(batch, N, t, N_CHANNELS)
            # moving data to platform
            data = data.to(platf)
            video_labels = video_labels.to(platf)
            # predicting logits
            logits = model(data)
            total = total + BATCH_SIZE_MODEL
            manipulation_probs = torch.round(torch.sigmoid(logits))
            video_labels = video_labels.tolist()
            manipulation_probs = np.array(manipulation_probs)
            video_labels = np.array(video_labels)
            true_positives = np.sum(manipulation_probs == video_labels)
    # EVALUATION METRICS
    print_eval_metrics(test_labels, predictions_list, true_positives, total, net)


def main():
    # type of net
    if NET_TYPE == 'speednet':
        save_path = '/nas/home/smariani/video_interpolation/interp_cnn/models/speednet.pth'
        model = S3DG(num_classes=1, num_frames=T, input_channel=N_CHANNELS)
    else:  # re_net
        save_path = '/nas/home/smariani/video_interpolation/interp_cnn/models/re_net.pth'
        model = ReNet(n_frames=T, spatial_dim=N, in_channels=N_CHANNELS)
    # GPU parameters
    set_gpu(0)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    set_backend()
    set_seed()
    platf = platform()
    # Model Parameters
    # model.load_state_dict(torch.load(save_path))
    # model = nn.DataParallel(model)  # just to train faster (multi GPU)
    model.to(platf)
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-02)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, mode='min', verbose=True)
    epochs = 10
    best_acc = 0
    no_improvement = 0     # n of epochs with no improvements
    patience = 5          # max n of epoch with no improvements
    min_val_loss = np.inf
    history = []

    # TRAINING
    train_dl, test_dl, valid_dl = load_data(BATCH_SIZE_LOAD)
    print("TRAINING " + NET_TYPE.upper())
    for e in range(epochs):
        print("EPOCH "+str(e+1))
        model.train()
        training(optimizer, criterion, model, train_dl, platf)
        # VALIDATING
        model.eval()
        correct, val_loss = validating(criterion, model, valid_dl, platf)
        lr_scheduler.step(val_loss)
        history.append({"epoch": e, "loss": val_loss, "lr": optimizer.param_groups[0]['lr']})
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
            df.to_csv("/nas/home/smariani/video_interpolation/interp_cnn/eval_metrics/" + NET_TYPE + "_history.csv")
            break

    print("Done!")
    # Save history
    df = pd.DataFrame(history)
    df.to_csv("/nas/home/smariani/video_interpolation/interp_cnn/eval_metrics/" + NET_TYPE + "_history.csv")

    # TESTING
    print("TESTING " + NET_TYPE.upper())
    model.eval()
    testing(model, test_dl, platf, T, NET_TYPE)


if __name__ == '__main__':
    main()
