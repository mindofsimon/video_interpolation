import torch
import torch.nn as nn
# from networks.speednet_torch import S3DG
from networks.global_speednet_torch import S3DG
# from interp_cnn_utils import *
from global_interp_cnn_utils import *
import pandas as pd
from utils.torch_utils import *
from tqdm import tqdm
from init_global_weights import global_state_dict
import pickle
from networks.linear_model import LinearModel
import torch.optim as optim


# Input Parameters
BATCH_SIZE_LOAD = 2  # how many original videos are extracted from the dataloaders at a time
BATCH_SIZE_MODEL = 2 * BATCH_SIZE_LOAD
BATCH_SIZE_LINEAR = 1  # how many elements are fed into the model at a time (original + interpolated)
# so in this way we extract BATCH_SIZE_LOAD original videos from the dataloaders at a time.
# then we load the interpolated version of those extracted videos, so we got a total of 2*BATCH_SIZE_LOAD videos.
# finally we fed those videos into the model.
N_CHANNELS = 1  # number of channels per frame (with optical flow is 3, with residuals is 1)
T = 16  # frames number
N = 224  # frames size (N x N)
MULTI_GPU_TRAINING = False  # if we want to train with double GPU
NEW_TRAINING_CYCLE = True  # if a new training starts or instead if we continue from a previous checkpoint


def training_linear(scores, labels, optimizer, criterion, model, platf):
    for i in tqdm(range(len(scores)), total=len(scores), desc='training'):
        # calculating loss and optimizing
        scores_tensor = torch.tensor(scores[i])
        labels_tensor = torch.tensor([labels[i]])
        scores_tensor = scores_tensor.to(platf)
        labels_tensor = labels_tensor.to(platf)
        model_output = model(scores_tensor)
        loss = criterion(model_output, labels_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validating_linear(scores, labels, criterion, model, platf):
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for i in tqdm(range(len(scores)), total=len(scores), desc='validating'):
            scores_tensor = torch.tensor(scores[i])
            labels_tensor = torch.tensor([labels[i]])
            scores_tensor = scores_tensor.to(platf)
            labels_tensor = labels_tensor.to(platf)
            model_output = model(scores_tensor)
            # calculating loss
            val_loss += criterion(model_output, labels_tensor).item()
            video_classes = torch.round(torch.sigmoid(model_output))
            video_classes = video_classes.tolist()
            video_classes = np.array(video_classes)
            video_labels = labels_tensor.tolist()
            video_labels = np.array(video_labels)
            correct += np.sum(video_classes == video_labels)
        val_loss /= (len(scores))
        # correct /= (len(valid_dl.dataset) * 2)
        correct /= (len(scores))
        print(f"validation error: \n accuracy: {(100 * correct):>0.1f}%, avg loss: {val_loss:>8f}")
    return correct, val_loss


def testing_linear():
    # TESTING
    with open('scores/test_scores.pkl', 'rb') as f1:
        scores = pickle.load(f1)
    with open('scores/test_labels.pkl', 'rb') as f2:
        labels = pickle.load(f2)
    save_path = '/nas/home/smariani/video_interpolation/interp_cnn/models/linear_layer.pth'
    model = LinearModel()
    # GPU parameters
    if MULTI_GPU_TRAINING:
        model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(save_path).items()})
    else:
        model.load_state_dict(torch.load(save_path))
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
    set_gpu(2)
    set_backend()
    set_seed()
    platf = platform()
    if MULTI_GPU_TRAINING:
        model = nn.DataParallel(model)  # just to train faster (multi GPU)
    model = model.to(platf)
    model.eval()
    # Accuracy Parameters
    true_positives = 0
    total = 0
    all_video_classes = []
    all_video_labels = []

    with torch.no_grad():
        for i in tqdm(range(len(scores)), total=len(scores), desc='testing'):
            scores_tensor = torch.tensor(scores[i])
            labels_tensor = torch.tensor([labels[i]])
            scores_tensor = scores_tensor.to(platf)
            labels_tensor = labels_tensor.to(platf)
            model_output = model(scores_tensor)
            total += BATCH_SIZE_LINEAR
            video_classes = torch.round(torch.sigmoid(model_output))
            video_classes = video_classes.tolist()
            # flat_list_video_classes = [item for sublist in video_classes for item in sublist]
            # all_video_classes = all_video_classes + flat_list_video_classes
            all_video_classes = all_video_classes + video_classes
            video_classes = np.array(video_classes)
            video_labels = labels_tensor.tolist()
            # flat_list_video_labels = [item for sublist in video_labels for item in sublist]
            # all_video_labels = all_video_labels + flat_list_video_labels
            all_video_labels = all_video_labels + video_labels
            video_labels = np.array(video_labels)
            true_positives += np.sum(video_classes == video_labels)
    # EVALUATION METRICS
    print_eval_metrics(all_video_labels, all_video_classes, true_positives, total)


def train_lin_layer():
    with open('scores/train_scores.pkl', 'rb') as f1:
        train_scores = pickle.load(f1)
    with open('scores/train_labels.pkl', 'rb') as f2:
        train_labels = pickle.load(f2)
    with open('scores/valid_scores.pkl', 'rb') as f3:
        valid_scores = pickle.load(f3)
    with open('scores/valid_labels.pkl', 'rb') as f4:
        valid_labels = pickle.load(f4)
    save_path = '/nas/home/smariani/video_interpolation/interp_cnn/models/linear_layer.pth'
    model = LinearModel()
    # GPU parameters
    if MULTI_GPU_TRAINING:
        os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
    else:
        set_gpu(2)
    set_backend()
    set_seed()
    platf = platform()
    if MULTI_GPU_TRAINING:
        model = nn.DataParallel(model)  # just to train faster (multi GPU)
    model = model.to(platf)
    criterion = nn.BCEWithLogitsLoss()
    # optimizer = optim.Adam(model.parameters(), lr=1e-02)
    optimizer = optim.Adam(model.parameters(), lr=1e-02)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, mode='min', verbose=True)
    epochs = 15
    best_acc = 0
    no_improvement = 0  # n of epochs with no improvements
    patience = 7  # max n of epoch with no improvements
    min_val_loss = np.inf
    history = []
    print("TRAINING LINEAR LAYER")
    for e in range(epochs):
        print("EPOCH " + str(e + 1))
        model.train()
        training_linear(train_scores, train_labels, optimizer, criterion, model, platf)
        # VALIDATING
        model.eval()
        correct, val_loss = validating_linear(valid_scores, valid_labels, criterion, model, platf)
        lr_scheduler.step(val_loss)
        history.append({"epoch": e, "accuracy": 100 * correct, "loss": val_loss, "lr": optimizer.param_groups[0]['lr']})
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
            df.to_csv("/nas/home/smariani/video_interpolation/interp_cnn/eval_metrics/linear_layer_history.csv")
            break

    print("Done!")
    # Save history
    df = pd.DataFrame(history)
    df.to_csv("/nas/home/smariani/video_interpolation/interp_cnn/eval_metrics/linear_layer_history.csv")



def save_scores():
    train_scores = []
    train_labels = []
    valid_scores = []
    valid_labels = []
    test_scores = []
    test_labels = []
    save_path = '/nas/home/smariani/video_interpolation/interp_cnn/models/speednet_global_2.pth'
    model = S3DG(num_classes=1, num_frames=T, input_channels=N_CHANNELS)
    # GPU parameters
    if MULTI_GPU_TRAINING:
        os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
    else:
        set_gpu(2)
    set_backend()
    set_seed()
    platf = platform()
    # Model Parameters
    pretrained_dict = global_state_dict(platf)
    model.load_state_dict(pretrained_dict)
    if MULTI_GPU_TRAINING:
        model = nn.DataParallel(model)  # just to train faster (multi GPU)
    model = model.to(platf)
    model = model.eval()
    train_dl, test_dl, valid_dl = load_data(BATCH_SIZE_LOAD)
    for batch in tqdm(train_dl, total=len(train_dl), desc='training scores computation'):
        data, video_labels = train_data_processing(batch, T, N_CHANNELS)
        # moving data to platform
        data = data.to(platf)
        video_labels = video_labels.to(platf)
        # predicting logits
        logits = model(data)
        video_scores = logits.tolist()
        train_scores = train_scores + video_scores
        video_labels = video_labels.tolist()
        video_labels = [item for sublist in video_labels for item in sublist]
        train_labels = train_labels + video_labels
    with open('scores/train_scores.pkl', 'wb') as f1:
        pickle.dump(train_scores, f1)
    with open('scores/train_labels.pkl', 'wb') as f2:
        pickle.dump(train_labels, f2)

    with open('scores/train_scores.pkl', 'rb') as f3:
        scores = pickle.load(f3)
    with open('scores/train_labels.pkl', 'rb') as f4:
        labels = pickle.load(f4)

    print(len(scores), len(labels))


if __name__ == '__main__':
    #train_lin_layer()
    testing_linear()
