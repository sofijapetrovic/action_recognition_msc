import torch
from torch import nn as nn
import torch.nn.functional as F


class ActionBlock(nn.Module):
    def __init__(self, feature_num_multiplier=1.):
        super(ActionBlock, self).__init__()
        k = feature_num_multiplier
        self.conv7 = nn.Conv2d(int(224 * k), int(112 * k), kernel_size=(1, 1), stride=(1, 1)).cuda()
        self.relu7 = nn.ReLU(inplace=True).cuda()
        self.conv8 = nn.Conv2d(int(112 * k), int(224 * k), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).cuda()
        self.bn4 = nn.BatchNorm2d(int(224 * k), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.relu8 = nn.ReLU(inplace=True).cuda()
        self.conv9 = nn.Conv2d(int(224 * k), int(224 * k), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).cuda()
        self.bn5 = nn.BatchNorm2d(int(224 * k), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.relu9 = nn.ReLU(inplace=True).cuda()

        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False).cuda()
        self.conv10 = nn.Conv2d(int(224 * k), 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).cuda()
        self.sm = nn.Softmax().cuda()

        self.up = nn.Upsample(scale_factor=2)
        self.conv11 = nn.Conv2d(2, int(224 * k), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).cuda()

    def forward(self, input):
        out = self.conv7(input)
        out = self.relu7(out)
        out = self.conv8(out)
        out = self.relu8(out)
        out = torch.sum(torch.stack([out, input]), dim=0)
        branch1 = out
        out = self.bn4(out)
        out = self.conv9(out)
        branch2 = out
        out = self.relu9(out)
        out = self.bn5(out)
        out = self.mp2(out)
        out = self.conv10(out)
        '''global max pooling'''
        out_action = F.max_pool2d(out, kernel_size=out.size()[2:])
        out_action = self.sm(out_action)
        out = self.up(out)
        out = self.conv11(out)
        if out.shape[2] < branch1.shape[2]:
            out = torch.cat([out, torch.zeros((out.shape[0], out.shape[1], 1, out.shape[3])).cuda()], dim=2)
        if out.shape[3] < branch1.shape[3]:
            out = torch.cat([out, torch.zeros((out.shape[0], out.shape[1], out.shape[2], 1)).cuda()], dim=3)
        out = torch.sum(torch.stack([out, branch1]), dim=0) #[:, :, :out.shape[2], :out.shape[3]]]
        out = torch.sum(torch.stack([out, branch2]), dim=0)
        out_block = out
        return out_action, out_block


class ActionNetwork(nn.Module):
    def __init__(self, in_channels, feature_num_multiplier=1.):
        super(ActionNetwork, self).__init__()
        k = feature_num_multiplier
        self.conv1 = nn.Conv2d(in_channels, int(12 * k), kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)).cuda()
        self.relu1 = nn.ReLU(inplace=True).cuda()
        self.conv2 = nn.Conv2d(in_channels, int(24 * k), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).cuda()
        self.relu2 = nn.ReLU(inplace=True).cuda()
        self.conv3 = nn.Conv2d(in_channels, int(36 * k), kernel_size=(3, 5), stride=(1, 1), padding=(1, 2)).cuda()
        self.relu3 = nn.ReLU(inplace=True).cuda()
        self.bn1 = nn.BatchNorm2d(int(72 * k), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()

        self.conv4 = nn.Conv2d(int(72 * k), int(112 * k), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).cuda()
        self.relu4 = nn.ReLU(inplace=True).cuda()
        self.bn2 = nn.BatchNorm2d(int(112 * k), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.conv5 = nn.Conv2d(int(72 * k), int(64 * k), kernel_size=(1, 1), stride=(1, 1)).cuda()
        self.relu5 = nn.ReLU(inplace=True).cuda()
        self.conv6 = nn.Conv2d(int(64 * k), int(112 * k), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).cuda()
        self.relu6 = nn.ReLU(inplace=True).cuda()
        self.bn3 = nn.BatchNorm2d(int(112 * k), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False).cuda()
        self.action_stage1 = ActionBlock(k)
        self.action_stage2 = ActionBlock(k)

    def forward(self, input):
        out1 = self.conv1(input)
        out1 = self.relu1(out1)
        out2 = self.conv2(input)
        out2 = self.relu2(out2)
        out3 = self.conv3(input)
        out3 = self.relu3(out3)

        out = torch.cat((out1, out2, out3), 1)
        out = self.bn1(out)

        out1 = self.conv4(out)
        out1 = self.relu4(out1)
        out1 = self.bn2(out1)
        out2 = self.conv5(out)
        out2 = self.relu5(out2)
        out2 = self.conv6(out2)
        out2 = self.relu6(out2)
        out2 = self.bn3(out2)
        out = torch.cat((out1, out2), 1)
        out = self.mp1(out)
        out_action1, out_next = self.action_stage1(out)
        out_action2, out = self.action_stage2(out_next)
        return out_action1, out_action2


class ActionModel:
    def __init__(self):
        super(ActionModel, self).__init__()
        self.pose_action_network = ActionNetwork(2, feature_num_multiplier=1/2)
        self.appearance_action_network = ActionNetwork(512)
        self.sm = nn.Softmax().cuda()

    def forward(self, pose_input, app_input):
        appearance_out = self.appearance_action_network(app_input)
        pose_out = self.pose_action_network(pose_input)
        return pose_out, appearance_out