"""
You can use this function if you want to test SpeedNet on a single file.
The functions here have no documentations, they are just adaptation of the functions that you can find on speednet_utils.py
"""

from speednet import *


def preprocess_test_video_single_file(vid, n, t):
    f_list = []
    cap = cv2.VideoCapture(vid)
    success, frame = cap.read()
    while success:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res_frame_rgb = image_resize(frame_rgb, height=n)
        if res_frame_rgb.shape[1] < n:  # in case width is lower than 224 pixels, we resize it to 224
            res_frame_rgb = cv2.resize(res_frame_rgb, dsize=(n, n), interpolation=cv2.INTER_NEAREST)
        crop_frame_rgb = center_crop(res_frame_rgb)
        f_list.append(crop_frame_rgb/255)
        success, frame = cap.read()

    f_list = f_list[0:t]  # taking only T frames
    return f_list


def generate_data_single(f_list, n, t):
    skip_file = False
    data = None
    if len(f_list) < t:
        skip_file = True
    else:
        frames_list = np.array(f_list)
        data = torch.autograd.Variable(torch.tensor(frames_list))
        data = torch.reshape(data, (1, t, n, n, 3))
        data = torch.permute(data, [0, 4, 1, 2, 3])
        data = data.float()

    return data, skip_file


def test_val_data_processing_single_file(vid, n, t):
    frames_list = preprocess_test_video_single_file(vid, n, t)
    data, skip_file = generate_data_single(frames_list, n, t)
    return data, skip_file


def testing_single_file(mod, video, platf, t):
    # TESTING
    with torch.no_grad():
        data,  skip = test_val_data_processing(video, N, t)
        if not skip:
            # moving data to platform
            data = data.to(platf)
            # predicting logits
            logits = mod(data)
            manipulation_probability = torch.sigmoid(logits)

            # OUTPUT
            if manipulation_probability == 0.0:
                print("NORMAL-SPEED VIDEO")
            else:
                print("SPED-UP VIDEO")
        else:
            print("ERROR : VIDEO HAS TOO LOW NUMBER OF FRAMES!")


# Input Parameters
T = 16  # frame number

# GPU parameters
set_gpu(0)
set_backend()
set_seed()
plat = platform()

# Model Parameters
SAVE_PATH = 'speednet.pth'
video_path = ''  # put here the path to the video you want to test
model = S3DG(num_classes=1, num_frames=T)
model.load_state_dict(torch.load(SAVE_PATH))
model.to(plat)
model.eval()
testing(model, video_path, plat, T)
