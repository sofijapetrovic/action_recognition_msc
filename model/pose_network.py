import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import sys
import os

sys.path.append(os.getcwd())
from utils.setup_logging import setup_logging
from torchsummary import summary
from model.pose_network_blocks import StageBlock
from model.pose_backbone import VGGbackbone


class PoseNet(nn.Module):
    def __init__(self, out_paf_channels=12, out_keypoint_channels=7, paf_output=1,
                 output_shape=(224, 398), sigma=5, upscale=True):
        super(PoseNet, self).__init__()
        self.paf_output = paf_output
        self.backbone = VGGbackbone()
        self.paf_stage1 = StageBlock(512, out_channels=out_paf_channels, conv_channels=32, convb_channels=64)
        self.heat_stage1 = StageBlock(512 + out_paf_channels, out_channels=out_keypoint_channels, conv_channels=32,
                                      convb_channels=64)
        self.output_shape = output_shape
        self.sigma = sigma
        self.upscale = upscale

    def forward(self, x):
        out = self.backbone(x)
        F = out
        out, paf_out = self.paf_stage1(out)
        if self.upscale:
            up = nn.Upsample(size=self.output_shape, mode='bicubic')
            paf_out1 = up(paf_out).cuda()
        else:
            paf_out1 = paf_out.cuda()
        out = torch.cat((out, F), 1)
        out, heat_out = self.heat_stage1(out)
        if self.upscale:
            up = nn.Upsample(size=self.output_shape, mode='bicubic')
            heat_out1 = up(heat_out).cuda()
        else:
            heat_out1 = heat_out.cuda()
        return F, paf_out1, heat_out1

    def get_output_images(self, image_tensor, paf=None, heat=None):
        """
        Creates 2D images out of the paf and heat layers
        Args:
            image_tensor ():

        Returns:
        """
        if paf is None or heat is None:
            out = self(torch.unsqueeze(image_tensor, 0).cuda())
            paf = out[self.paf_output][0, :, :, :]
            heat = out[-1][0, :, :, :]
        else:
            paf = paf[0, :, :, :]
            heat = heat[0, :, :, :]
        paf = torch.sum(torch.abs(paf), dim=0)
        paf = torch.stack([paf, paf, paf], dim=0)
        heat = torch.sum(torch.clip(heat, 0, 1), dim=0)
        heat = torch.stack([heat, heat, heat], dim=0)
        return image_tensor.cpu(), paf.cpu(), heat.cpu()


if __name__ == '__main__':
    setup_logging()
    model = PoseNet()
    summary(model, (3, 288, 512))
    dummy = torch.randn((1, 3, 288, 512)).cuda().detach()
    writer = SummaryWriter(log_dir='../logs/')
    writer.add_graph(model, input_to_model=dummy)
    writer.flush()
    writer.close()
