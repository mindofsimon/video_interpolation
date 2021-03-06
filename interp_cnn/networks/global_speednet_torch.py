import torch.nn as nn
import torch


class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=False)

        self.bn = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class STConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(STConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=(1, kernel_size, kernel_size),
                              stride=(1, stride, stride), padding=(0, padding, padding))
        self.conv2 = nn.Conv3d(out_planes, out_planes, kernel_size=(kernel_size, 1, 1),
                               stride=(1, 1, 1), padding=(padding, 0, 0))

        self.bn = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu = nn.ReLU(inplace=True)

        self.bn2 = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class Mixed_3b(nn.Module):
    def __init__(self):
        super(Mixed_3b, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(192, 64, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(192, 96, kernel_size=1, stride=1),
            STConv3d(96, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(192, 16, kernel_size=1, stride=1),
            STConv3d(16, 32, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(192, 32, kernel_size=1, stride=1),
        )

        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Conv3d(1, 1, (1, 1, 1), stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)

        temp = out
        out = self.squeeze(out)
        out = self.excitation(out.permute(0, 2, 1, 3, 4))
        out = self.sigmoid(out)

        return out.permute(0, 2, 1, 3, 4) * temp


class Mixed_3c(nn.Module):
    def __init__(self):
        super(Mixed_3c, self).__init__()
        self.branch0 = nn.Sequential(
            BasicConv3d(256, 128, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(256, 128, kernel_size=1, stride=1),
            STConv3d(128, 192, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(256, 32, kernel_size=1, stride=1),
            STConv3d(32, 96, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(256, 64, kernel_size=1, stride=1),
        )

        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Conv3d(1, 1, (1, 1, 1), stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)

        temp = out
        out = self.squeeze(out)
        out = self.excitation(out.permute(0, 2, 1, 3, 4))
        out = self.sigmoid(out)

        return out.permute(0, 2, 1, 3, 4) * temp


class Mixed_4b(nn.Module):
    def __init__(self):
        super(Mixed_4b, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(480, 192, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(480, 96, kernel_size=1, stride=1),
            STConv3d(96, 208, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(480, 16, kernel_size=1, stride=1),
            STConv3d(16, 48, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(480, 64, kernel_size=1, stride=1),
        )

        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Conv3d(1, 1, (1, 1, 1), stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)

        temp = out
        out = self.squeeze(out)
        out = self.excitation(out.permute(0, 2, 1, 3, 4))
        out = self.sigmoid(out)

        return out.permute(0, 2, 1, 3, 4) * temp


class Mixed_4c(nn.Module):
    def __init__(self):
        super(Mixed_4c, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(512, 160, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(512, 112, kernel_size=1, stride=1),
            STConv3d(112, 224, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(512, 24, kernel_size=1, stride=1),
            STConv3d(24, 64, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(512, 64, kernel_size=1, stride=1),
        )

        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Conv3d(1, 1, (1, 1, 1), stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)

        temp = out
        out = self.squeeze(out)
        out = self.excitation(out.permute(0, 2, 1, 3, 4))
        out = self.sigmoid(out)

        return out.permute(0, 2, 1, 3, 4) * temp


class Mixed_4d(nn.Module):
    def __init__(self):
        super(Mixed_4d, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(512, 128, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(512, 128, kernel_size=1, stride=1),
            STConv3d(128, 256, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(512, 24, kernel_size=1, stride=1),
            STConv3d(24, 64, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(512, 64, kernel_size=1, stride=1),
        )

        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Conv3d(1, 1, (1, 1, 1), stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)

        temp = out
        out = self.squeeze(out)
        out = self.excitation(out.permute(0, 2, 1, 3, 4))
        out = self.sigmoid(out)

        return out.permute(0, 2, 1, 3, 4) * temp


class Mixed_4e(nn.Module):
    def __init__(self):
        super(Mixed_4e, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(512, 112, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(512, 144, kernel_size=1, stride=1),
            STConv3d(144, 288, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(512, 32, kernel_size=1, stride=1),
            STConv3d(32, 64, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(512, 64, kernel_size=1, stride=1),
        )

        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Conv3d(1, 1, (1, 1, 1), stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)

        temp = out
        out = self.squeeze(out)
        out = self.excitation(out.permute(0, 2, 1, 3, 4))
        out = self.sigmoid(out)

        return out.permute(0, 2, 1, 3, 4) * temp


class Mixed_4f(nn.Module):
    def __init__(self):
        super(Mixed_4f, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(528, 256, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(528, 160, kernel_size=1, stride=1),
            STConv3d(160, 320, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(528, 32, kernel_size=1, stride=1),
            STConv3d(32, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(528, 128, kernel_size=1, stride=1),
        )

        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Conv3d(1, 1, (1, 1, 1), stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)

        temp = out
        out = self.squeeze(out)
        out = self.excitation(out.permute(0, 2, 1, 3, 4))
        out = self.sigmoid(out)

        return out.permute(0, 2, 1, 3, 4) * temp


class Mixed_5b(nn.Module):
    def __init__(self):
        super(Mixed_5b, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(832, 256, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(832, 160, kernel_size=1, stride=1),
            STConv3d(160, 320, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(832, 32, kernel_size=1, stride=1),
            STConv3d(32, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(832, 128, kernel_size=1, stride=1),
        )

        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Conv3d(1, 1, (1, 1, 1), stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)

        temp = out
        out = self.squeeze(out)
        out = self.excitation(out.permute(0, 2, 1, 3, 4))
        out = self.sigmoid(out)

        return out.permute(0, 2, 1, 3, 4) * temp


class Mixed_5c(nn.Module):
    def __init__(self):
        super(Mixed_5c, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(832, 384, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(832, 192, kernel_size=1, stride=1),
            STConv3d(192, 384, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(832, 48, kernel_size=1, stride=1),
            STConv3d(48, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(832, 128, kernel_size=1, stride=1),
        )

        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Conv3d(1, 1, (1, 1, 1), stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)

        temp = out
        out = self.squeeze(out)
        out = self.excitation(out.permute(0, 2, 1, 3, 4))
        out = self.sigmoid(out)
        return out.permute(0, 2, 1, 3, 4) * temp


class S3DG(nn.Module):

    def __init__(self, num_classes=1, num_frames=64, input_channels=3):
        super(S3DG, self).__init__()
        self.frames = nn.Sequential(
            STConv3d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            BasicConv3d(64, 64, kernel_size=1, stride=1),
            STConv3d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            Mixed_3b(),
            Mixed_3c(),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            Mixed_4b(),
            Mixed_4c(),
            Mixed_4d(),
            Mixed_4e(),
            Mixed_4f(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0)),
            Mixed_5b(),
            Mixed_5c(),
            nn.MaxPool3d(kernel_size=(num_frames, 1, 1), stride=1),  # equal to global max pooling (reducing space)
            nn.AdaptiveAvgPool3d(output_size=(1, 1, 1)),  # equal to global average pooling (reducing time)
            nn.Conv3d(1024, 512, kernel_size=1, stride=1, bias=True)
        )
        self.residuals = nn.Sequential(
            STConv3d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            BasicConv3d(64, 64, kernel_size=1, stride=1),
            STConv3d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            Mixed_3b(),
            Mixed_3c(),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            Mixed_4b(),
            Mixed_4c(),
            Mixed_4d(),
            Mixed_4e(),
            Mixed_4f(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0)),
            Mixed_5b(),
            Mixed_5c(),
            nn.MaxPool3d(kernel_size=(num_frames, 1, 1), stride=1),  # equal to global max pooling (reducing space)
            nn.AdaptiveAvgPool3d(output_size=(1, 1, 1)),  # equal to global average pooling (reducing time)
            nn.Conv3d(1024, 512, kernel_size=1, stride=1, bias=True)
        )
        self.of = nn.Sequential(
            STConv3d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            BasicConv3d(64, 64, kernel_size=1, stride=1),
            STConv3d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            Mixed_3b(),
            Mixed_3c(),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            Mixed_4b(),
            Mixed_4c(),
            Mixed_4d(),
            Mixed_4e(),
            Mixed_4f(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0)),
            Mixed_5b(),
            Mixed_5c(),
            nn.MaxPool3d(kernel_size=(num_frames, 1, 1), stride=1),  # equal to global max pooling (reducing space)
            nn.AdaptiveAvgPool3d(output_size=(1, 1, 1)),  # equal to global average pooling (reducing time)
            nn.Conv3d(1024, 512, kernel_size=1, stride=1, bias=True)
        )
        self.dropout_1 = nn.Dropout(0.5)
        self.dropout_2 = nn.Dropout(0.5)
        self.dropout_3 = nn.Dropout(0.5)
        self.linear_1 = nn.Linear(512, num_classes)
        self.linear_2 = nn.Linear(512, num_classes)
        self.linear_3 = nn.Linear(512, num_classes)
        self.final_linear = nn.Linear(3, 1)
        # nn.init.xavier_uniform_(self.final_linear.weight)

    def forward(self, x):  # [2, 3, 1, 16, 224, 224]
        # input x is BATCH_SIZE x [frames, res, of] x CHANNELS x N_FRAMES x H x W
        x_1 = x[:, 0]  # frames [2, 1, 16, 224, 224]
        x_1 = self.frames(x_1)
        x_1 = self.dropout_1(x_1.view(x_1.size(0), -1))
        x_1 = self.linear_1(x_1)
        x_2 = x[:, 1]  # residuals [2, 1, 16, 224, 224]
        x_2 = self.residuals(x_2)
        x_2 = self.dropout_2(x_2.view(x_2.size(0), -1))
        x_2 = self.linear_2(x_2)
        x_3 = x[:, 2]  # optical flow [2, 1, 16, 224, 224]
        x_3 = self.of(x_3)
        x_3 = self.dropout_3(x_3.view(x_3.size(0), -1))
        x_3 = self.linear_3(x_3)
        x_tot = torch.cat((x_1, x_2, x_3), 1)  # concatenating outputs [2, 3]
        # x_tot = self.final_linear(x_tot)  # final output [2, 1]
        return x_tot  # sigmoid will be applied outside
