import torch.nn as nn


class LinearModel(nn.Module):

    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        out = self.linear(x)
        return out
