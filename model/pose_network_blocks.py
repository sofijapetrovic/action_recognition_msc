import torch
import torch.nn as nn
from utils.setup_logging import setup_logging
from torchsummary import summary


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=32):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).cuda()
        self.bn1 = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.relu1 = nn.ReLU(inplace=True).cuda()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).cuda()
        self.bn2 = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.relu2 = nn.ReLU(inplace=True).cuda()
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).cuda()
        self.bn3 = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.relu3 = nn.ReLU(inplace=True).cuda()

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.relu1(out1)

        out2 = self.conv2(out1)
        out2 = self.bn2(out2)
        out2 = self.relu2(out2)

        out3 = self.conv3(out2)
        out3 = self.bn3(out3)
        out3 = self.relu3(out3)
        out = torch.cat((out1, out2, out3), 1)
        return out


class StageBlock(nn.Module):
    def __init__(self, in_channels, out_channels, convb_channels, conv_channels):
        super(StageBlock, self).__init__()
        self.convb1 = ConvBlock(in_channels=in_channels, out_channels=convb_channels)
        self.convb2 = ConvBlock(in_channels=3*convb_channels, out_channels=convb_channels)
        self.convb3 = ConvBlock(in_channels=3*convb_channels, out_channels=convb_channels)
        self.convb4 = ConvBlock(in_channels=3*convb_channels, out_channels=convb_channels)
        self.convb5 = ConvBlock(in_channels=3*convb_channels, out_channels=convb_channels)
        self.conv1 = nn.Conv2d(3*convb_channels, conv_channels, kernel_size=(1, 1), stride=(1, 1)).cuda()
        self.relu1 = nn.ReLU(inplace=True).cuda()
        self.conv2 = nn.Conv2d(conv_channels, out_channels, kernel_size=(1, 1), stride=(1, 1)).cuda()

    def forward(self, x):
        out = self.convb1(x)
        out = self.convb2(out)
        out = self.convb3(out)
        out = self.convb4(out)
        out = self.convb5(out)
        out = self.conv1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        return out, out


if __name__ == '__main__':
    setup_logging()
    conv = StageBlock(in_channels=512, out_channels=12, convb_channels=32, conv_channels=128)
    summary(conv, (512,224,224))