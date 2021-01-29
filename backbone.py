import torch
import logging
import torch.nn as nn
import logging
from utils import setup_logging
from torchvision import transforms


class VGGbackbone(nn.Module):
    def __init__(self):
        super(VGGbackbone, self).__init__()
        self.process = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).cuda()
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.relu1 = nn.ReLU(inplace=True).cuda()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).cuda()
        self.bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.relu2 = nn.ReLU(inplace=True).cuda()
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False).cuda()
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).cuda()
        self.bn3 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.relu3 = nn.ReLU(inplace=True).cuda()
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).cuda()
        self.bn4 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.relu4 = nn.ReLU(inplace=True).cuda()
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False).cuda()
        self.conv5 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).cuda()
        self.bn5 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.relu5 = nn.ReLU(inplace=True).cuda()
        self.conv6 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).cuda()
        self.bn6 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.relu6 = nn.ReLU(inplace=True).cuda()
        self.conv7 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).cuda()
        self.bn7 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.relu7 = nn.ReLU(inplace=True).cuda()
        self.conv8 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).cuda()
        self.bn8 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.relu8 = nn.ReLU(inplace=True).cuda()
        self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False).cuda()
        self.conv9 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).cuda()
        self.bn9 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.relu9 = nn.ReLU(inplace=True).cuda()
        self.conv10 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).cuda()
        self.bn10 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.relu10 = nn.ReLU(inplace=True).cuda()
        self.load_pretrained_weights()

    def forward(self, x):
        x = self.process(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.mp1(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu4(out)
        out = self.mp2(out)
        out = self.conv5(out)
        out = self.bn5(out)
        out = self.relu5(out)
        out = self.conv6(out)
        out = self.bn6(out)
        out = self.relu6(out)
        out = self.conv7(out)
        out = self.bn7(out)
        out = self.relu7(out)
        out = self.conv8(out)
        out = self.bn8(out)
        out = self.relu8(out)
        out = self.mp3(out)
        out = self.conv9(out)
        out = self.bn9(out)
        out = self.relu9(out)
        out = self.conv10(out)
        out = self.bn10(out)
        out = self.relu10(out)
        return out

    def load_pretrained_weights(self):
        pretrainedVGG19 = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19_bn', pretrained=True)
        VGG19_dict = pretrainedVGG19.state_dict()
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in VGG19_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        logging.info('Loaded VGG19 backbone weights')
        #print(self)

if __name__ == '__main__':
    setup_logging()
    back = VGGbackbone()

