"""
Complete SpeedNet implementation.
It includes Training, Validation and Testing.
The model takes as input a tensor of size [BATCH_SIZE, N_CHANNELS, N_FRAMES, HEIGHT, WIDTH].

N_CHANNELS is always 3.
N_FRAMES should be 64.
HEIGHT and WIDTH are both set to 224 pixels.
BATCH SIZE is 2 for training (we train with a double batch composed by an original video and its manipulated version) and 1 for test and validation.

Input videos are loaded through data loaders (check load_data.py and speednet_utils.py for more details).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from s3dg_torch import S3DG
from speednet_utils import *
import numpy as np
import pandas as pd
from torch_utils import *


# Input Parameters
T = 64  # frame number
N = 224  # frame size (N x N)
SAVE_PATH = 'speednet.pth'  # location of model


def training(optimizer, criterion, model, train_dl, platf):
    """
    Training function
    :param optimizer: model optimizer
    :param criterion: model loss function
    :param model: actual model (S3DG)
    :param train_dl: train data loader
    :param platf : data will be loaded on this platform
    :return: nothing
    """
    count = 0
    for batch in train_dl:
        print("file "+str(count+1))
        count += 1
        # getting files path and label (each batch will contain original and manipulated versions of same video)
        # video 1 is original (class=0.0)
        # video 2 is manipulated (class=1.0)
        video_path, video_label, _ = batch
        video_path_1 = video_path[0]
        video_path_2 = video_path[1]
        video_label_1 = float(video_label[0])
        video_label_2 = float(video_label[1])
        # building input tensor
        frames_list_1 = preprocess_train_video(video_path_1, video_label_1, T, N)
        frames_list_2 = preprocess_train_video(video_path_2, video_label_2, T, N)
        frames_list = np.array([frames_list_1, frames_list_2])
        data = torch.autograd.Variable(torch.tensor(frames_list, dtype=float))
        data = torch.reshape(data, (2, T, N, N, 3))
        data = torch.permute(data, [0, 4, 1, 2, 3])
        data = data.float()
        data = data.to(platf)
        video_labels = torch.tensor([[video_label_1], [video_label_2]])
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
    :param model: actual model (S3DG)
    :param valid_dl: validation data loader
    :param platf : data will be loaded on this platform
    :return: nothing
    """
    # VALIDATION
    val_loss = 0
    correct = 0
    val_count = 0
    model.eval()
    print("START VALIDATION")
    with torch.no_grad():
        for batch in valid_dl:
            video_path, video_label, _ = batch
            video_path = video_path[0]
            video_label = float(video_label[0])
            frames_list = preprocess_test_video(video_path, N, T)
            frames_list = np.array([frames_list])
            data = torch.autograd.Variable(torch.tensor(frames_list, dtype=float))
            data = torch.reshape(data, (1, T, N, N, 3))
            data = torch.permute(data, [0, 4, 1, 2, 3])
            data = data.float()
            data = data.to(platf)
            video_label = torch.tensor([[video_label]])
            video_label = video_label.to(platf)
            logits = model(data)
            val_loss += criterion(logits, video_label).item()
            if torch.round(torch.sigmoid(logits)) == video_label:
                correct += 1
                val_count += 1

        print("END VALIDATION")
        val_loss /= 1
        correct /= val_count
    return val_loss, correct


def testing(model, test_dl, platf):
    """
    Test function
    :param model: actual model (S3DG)
    :param test_dl: test data loader
    :param platf : data will be loaded on this platform
    :return: nothing
    """
    # RELOAD MODEL AND TEST
    # Accuracy Parameters
    true_positives = 0
    total = 0
    test_labels = []
    predictions_list = []

    print("START TESTING")
    with torch.no_grad():
        for batch in test_dl:
            # getting file path and label
            video_path, video_label, _ = batch
            video_path = video_path[0]
            video_label = float(video_label[0])
            # building input tensor
            frames_list = preprocess_test_video(video_path, N, T)
            frames_list = np.array([frames_list])
            data = torch.autograd.Variable(torch.tensor(frames_list, dtype=float))
            data = torch.reshape(data, (1, T, N, N, 3))
            data = torch.permute(data, [0, 4, 1, 2, 3])
            data = data.float()
            # predicting logits and converting into probability
            data = data.to(platf)
            logits = model(data)
            manipulation_probability = torch.sigmoid(logits)
            total = total + 1
            predictions_list.append(np.round(manipulation_probability[0].item()))
            test_labels.append(video_label)
            if torch.round(manipulation_probability) == video_label:
                true_positives = true_positives + 1
        print("END TESTING")
    # EVALUATION METRICS
    print_eval_metrics(test_labels, predictions_list, true_positives, total)


def main():
    # GPU parameters
    set_gpu(0)
    set_backend()
    set_seed()
    platf = platform()
    # Model Parameters
    model = S3DG(num_classes=1, num_frames=T)
    model.to(platf)
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, mode='min', verbose=True)
    epochs = 10
    best_acc = 0
    no_improvement = 0     # n of epochs with no improvements
    patience = 10          # max n of epoch with no improvements
    min_val_loss = np.inf
    history = []

    # TRAINING
    train_dl, test_dl, valid_dl = load_data()
    print("START TRANINING")
    for e in range(epochs):
        print("EPOCH "+str(e+1))
        training(optimizer, criterion, model, train_dl, platf)
        # VALIDATING
        correct, val_loss = validating(criterion, model, valid_dl, platf)
        lr_scheduler.step(val_loss)
        model.train()
        history.append({"epoch": e, "loss": val_loss, "lr": optimizer.param_groups[0]['lr']})
        # MODEL CHECKPOINT CALLBACK
        accuracy = correct
        if accuracy > best_acc:
            # Callback for weight saving
            torch.save(model.state_dict(), SAVE_PATH)
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
            df.to_csv("history.csv")
            break

    print("Done!")
    # Save history
    df = pd.DataFrame(history)
    df.to_csv("history.csv")

    # SAVING TRAINED MODEL
    torch.save(model.state_dict(), SAVE_PATH)

    # TESTING
    # Model Parameters
    # model = S3DG(num_classes=1, num_frames=T)
    # model.load_state_dict(torch.load(SAVE_PATH))
    model.eval()
    testing(model, test_dl, platf)


if __name__ == '__main__':
    main()
