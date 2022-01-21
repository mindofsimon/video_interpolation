"""
Network based on CNN Rescaling Factor Estimation by Chang Liu and Matthias Kirchner.
The network has been adapted to take segments of video frames instead of individual images.
Network input is 5D : [BATCH_SIZE, N_CHANNELS, N_FRAMES, HEIGHT, WIDTH].
"""
import torch.nn as nn


class ReNet(nn.Module):
    def __init__(self, n_frames, spatial_dim, in_channels):
        super(ReNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 16, (1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),

            nn.Conv3d(16, 16, (1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.Conv3d(16, 16, (1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.Conv3d(16, 16, (1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.Conv3d(16, 16, (1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.Conv3d(16, 16, (1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.Conv3d(16, 16, (1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.Conv3d(16, 16, (1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.Conv3d(16, 16, (1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.Conv3d(16, 16, (1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.Conv3d(16, 16, (1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.Conv3d(16, 16, (1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.Conv3d(16, 16, (1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.Conv3d(16, 16, (1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.Conv3d(16, 16, (1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.Conv3d(16, 16, (1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.Conv3d(16, 16, (1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.Conv3d(16, 16, (1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.Conv3d(16, 16, (1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.Conv3d(16, 16, (1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.Conv3d(16, 16, (1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.Conv3d(16, 16, (1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.Conv3d(16, 16, (1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.Conv3d(16, 16, (1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.Conv3d(16, 16, (1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),

            nn.MaxPool3d((1, 1, 1), padding=(0, 0, 0), stride=(1, 4, 4)),

            nn.Flatten(1, 4),
            nn.Linear((16*n_frames*int(spatial_dim/4)*int(spatial_dim/4)), 1)
        )

    def forward(self, x):
        logits = self.features(x)
        return logits
