import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from utils import setup_logging
from torchsummary import summary
from model_blocks import StageBlock
from backbone import VGGbackbone


class PoseNet(nn.Module):
    def __init__(self, out_paf_channels=12, out_heat_channels=7, paf_stages=2, heat_stages=2, output_shape=(224, 398)):
        super(PoseNet, self).__init__()
        self.paf_stages = paf_stages
        self.heat_stages = heat_stages
        self.backbone = VGGbackbone()
        self.paf_stage1 = StageBlock(512,out_channels=out_paf_channels,conv_channels=32,convb_channels=64)
        self.paf_stage2 = StageBlock(out_paf_channels,out_channels=out_paf_channels,conv_channels=32,convb_channels=64)
        self.paf_stage3 = StageBlock(out_paf_channels, out_channels=out_paf_channels, conv_channels=32,
                                     convb_channels=64)
        self.paf_stage4 = StageBlock(out_paf_channels, out_channels=out_paf_channels, conv_channels=32,
                                     convb_channels=64)
        self.heat_stage1 = StageBlock(512+out_paf_channels,out_channels=out_heat_channels,conv_channels=32,convb_channels=64)
        self.heat_stage2 = StageBlock(out_heat_channels,out_channels=out_heat_channels,conv_channels=32,convb_channels=64)
        self.output_shape = output_shape

    def forward(self, x):
        out = self.backbone(x)
        F = out
        out, paf_out = self.paf_stage1(out)
        up = nn.Upsample(size=self.output_shape, mode='bicubic')
        paf_out1 = up(paf_out).cuda()
        out, paf_out = self.paf_stage2(out)
        up = nn.Upsample(size=self.output_shape, mode='bicubic')
        paf_out2 = up(paf_out).cuda()
        #out, paf_out = self.paf_stage2(out)
        #up = nn.Upsample(size=self.output_shape, mode='bicubic')
        #paf_out3 = up(paf_out).cuda()

        out = torch.cat((out, F), 1)
        out, heat_out = self.heat_stage1(out)
        up = nn.Upsample(size=self.output_shape, mode='bicubic')
        heat_out1 = up(heat_out).cuda()
        out, heat_out = self.heat_stage2(out)
        up = nn.Upsample(size=self.output_shape, mode='bicubic')
        heat_out2 = up(heat_out)
        return paf_out1,paf_out2,heat_out1,heat_out2

    def get_output_images(self,image_tensor):
        out = self(torch.unsqueeze(image_tensor, 0).cuda())
        paf = out[self.paf_stages-1][0,:,:,:]
        heat = out[-1][0,:,:,:]
        paf = torch.sum(torch.abs(paf),dim=0)
        paf = torch.stack([paf,paf,paf],dim=0)
        heat = torch.sum(torch.clip(heat,0,1),dim=0)
        heat = torch.stack([heat,heat,heat],dim=0)
        return image_tensor.cpu(), paf.cpu(), heat.cpu()


if __name__ == '__main__':
    setup_logging()
    model = PoseNet()
    summary(model, (3, 288, 512))
    dummy =torch.randn((1,3,288,512)).cuda().detach()
    writer = SummaryWriter(log_dir='logs/')
    writer.add_graph(model, input_to_model=dummy)
    writer.flush()
    writer.close()



