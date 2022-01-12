"""
Complete SpeedNet implementation.
It includes Training, Validation and Testing.
The model takes as input a tensor of size [BATCH_SIZE, N_CHANNELS (3), N_FRAMES (T), HEIGHT (N), WIDTH (N)].

N_CHANNELS is always 3.
N_FRAMES (temporal dimension) is set to 64.
HEIGHT = WIDTH = N.
For test and validation videos N (spatial dimension) is set to 224.
For train videos N (spatial dimension) is set to a random int between 64 and 336.
BATCH SIZE is always 2 (we train/test/validate with a double batch composed by an original video and its manipulated version)

Input videos are loaded through data loaders (check load_data.py and speednet_utils.py for more details).
"""
import torch
import torch.nn as nn
import torch.optim as optim
from s3dg_torch import S3DG
from speednet_utils import *
import pandas as pd
from torch_utils import *
from tqdm import tqdm


# Input Parameters
T = 64  # frame number
N = 224  # frame size (N x N)
SAVE_PATH = '/nas/home/smariani/video_interpolation/speednet/speednet.pth'  # location of model


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

    for batch in tqdm(train_dl, total=len(train_dl), desc='training'):
        data, video_labels, skip = train_data_processing(batch, T)
        if not skip:
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
    :param model: actual model (S3DG)
    :param valid_dl: validation data loader
    :param platf : data will be loaded on this platform
    :return: nothing
    """

    # VALIDATION
    val_loss = 0
    correct = 0
    n_skipped = 0  # to count the number of skipped videos (due to low number of frames)
    with torch.no_grad():
        for batch in tqdm(valid_dl, total=len(valid_dl), desc='validating'):
            data, video_labels, skip = test_val_data_processing(batch, N, T)
            if not skip:
                # moving data to platform
                data = data.to(platf)
                video_labels = video_labels.to(platf)
                # predicting logits
                logits = model(data)
                # calculating loss
                val_loss += criterion(logits, video_labels).item()
                if torch.round(torch.sigmoid(logits))[0].item() == video_labels[0].item():
                    correct += 1
                if torch.round(torch.sigmoid(logits))[1].item() == video_labels[1].item():
                    correct += 1
            else:
                n_skipped += 1
        val_loss /= (len(valid_dl) - n_skipped)
        correct /= ((len(valid_dl.dataset) - n_skipped) * 2)
        print(f"validation error: \n accuracy: {(100 * correct):>0.1f}%, avg loss: {val_loss:>8f}")
    return correct, val_loss


def testing(model, test_dl, platf, t):
    """
    Test function
    :param model: actual model (S3DG)
    :param test_dl: test data loader
    :param platf : data will be loaded on this platform
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
            data, video_labels, skip = test_val_data_processing(batch, N, t)
            if not skip:
                # moving data to platform
                data = data.to(platf)
                video_labels = video_labels.to(platf)
                # predicting logits
                logits = model(data)
                manipulation_probabilities = torch.sigmoid(logits)
                total = total + 2
                predictions_list.append(np.round(manipulation_probabilities[0].item()))
                predictions_list.append(np.round(manipulation_probabilities[1].item()))
                test_labels.append(video_labels[0].item())
                test_labels.append(video_labels[1].item())
                if torch.round(manipulation_probabilities)[0].item() == video_labels[0].item():
                    true_positives = true_positives + 1
                if torch.round(manipulation_probabilities)[1].item() == video_labels[1].item():
                    true_positives = true_positives + 1
    # EVALUATION METRICS
    print_eval_metrics(test_labels, predictions_list, true_positives, total)


def main():
    # GPU parameters
    set_gpu(1)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    set_backend()
    set_seed()
    platf = platform()
    # Model Parameters
    model = S3DG(num_classes=1, num_frames=T)
    model.load_state_dict(torch.load(SAVE_PATH))
    # model = nn.DataParallel(model)  # just to train faster (multi GPU)
    model.to(platf)
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-02)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, mode='min', verbose=True)
    epochs = 1
    best_acc = 0
    no_improvement = 0     # n of epochs with no improvements
    patience = 5          # max n of epoch with no improvements
    min_val_loss = np.inf
    history = []

    # TRAINING
    train_dl, test_dl, valid_dl = load_data()
    print("TRAINING SPEEDNET")
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
            df.to_csv("/nas/home/smariani/video_interpolation/speednet/history.csv")
            break

    print("Done!")
    # Save history
    df = pd.DataFrame(history)
    df.to_csv("/nas/home/smariani/video_interpolation/speednet/history.csv")

    # TESTING (reloading model in order to test on segments of 16 frames)
    print("TESTING SPEEDNET")
    model = S3DG(num_classes=1, num_frames=16)
    model.load_state_dict(torch.load(SAVE_PATH))
    model.to(platf)
    model.eval()
    testing(model, test_dl, platf, 16)


if __name__ == '__main__':
    main()
