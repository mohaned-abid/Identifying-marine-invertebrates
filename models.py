import pretrainedmodels
import torch.nn as nn
from torch.nn import functional as F
class ResNet34(nn.Module):
    def __init__(self,pretrained):
        super(ResNet34,self).__init__()
        if pretrained is True:
            self.model=pretrainedmodels.__dict__['resnet34'](pretrained="imagenet")
        else:
            self.model=pretrainedmodels.__dict__['resnet34'](pretrained=None)
        self.l0=nn.Linear(512,137)
    def forward(self,x):
        bs=x.shape[0]
        x=self.model.features(x)
        x=F.adaptive_avg_pool2d(x,1).reshape(bs, -1)
        l0=self.l0(x)
        return l0