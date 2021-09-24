import numpy as np
import torch
from torch.nn import functional as F

from implem.utils import init_torch_device, as_tensor, device, dtype, dtype_np

from functools import partial
#from torchvision.models.resnet import BasicBlock # only needed when using ResNet architecture


def setup_conv(in_channels, out_channels, kernel_size, bias, padding_mode, stride=1, Conv=torch.nn.Conv2d):
    return Conv(in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=(kernel_size-1)//2,
                stride=stride,
                bias=bias)


class BilinearConvLayer(torch.nn.Module):

    def __init__(self,
                 input_channels,
                 output_channels,
                 bilin_channels=None,
                 padding_mode='zeros',
                 Conv=torch.nn.Conv2d,
                 nonlinearity=torch.nn.Identity(),
                 norm=torch.nn.Identity(),
                 kernel_size=3
                 ):

        super(BilinearConvLayer, self).__init__()


        # channel index cutoffs for grouping into bilinear layer inputs:
        bilin_channels = output_channels if bilin_channels is None else bilin_channels
        self.chgrp1 = max(0,output_channels-bilin_channels)
        self.chgrp2 = bilin_channels

        self.layer1 = setup_conv(in_channels = input_channels,
                                 out_channels = self.chgrp1 + 2*self.chgrp2,
                                 kernel_size = kernel_size,
                                 bias = True,
                                 padding_mode=padding_mode,
                                 stride=1,
                                 Conv=Conv)
        self.norm = norm
        self.nonlinearity = nonlinearity

    def forward(self, x):

        y = self.nonlinearity(self.norm(self.layer1(x)))

        # split tensor along channels: last two chunks go into bilinear layer
        mid = self.chgrp1+self.chgrp2
        y1, y2, y3 = y[:, :self.chgrp1], y[:, self.chgrp1:mid], y[:, mid:]

        # bilinear layer
        z = y2 * y3 # fixing x^t A y, A = torch.eye(C) for now

        out = torch.cat((y1, z), dim=1)

        return out


class BilinearConvNet(torch.nn.Module):

    def __init__(self, 
                 input_channels,
                 output_channels,
                 hidden_channels,
                 instance_dimensionality,
                 bilin_channels=None,
                 padding_mode='zeros',
                 normalization='BatchNorm',
                 nonlinearity='Identity'):

        super(BilinearConvNet, self).__init__()

        out_channels = hidden_channels + [output_channels]
        if bilin_channels is None or bilin_channels=='None':
            bilin_channels = [c//2 for c in out_channels]
        assert len(bilin_channels) == len(out_channels)

        if nonlinearity=='ReLU':
            nonlinearity = torch.nn.ReLU()
        elif nonlinearity=='Identity':
            nonlinearity = torch.nn.Identity()

        if normalization=='BatchNorm':
            Norm = torch.nn.BatchNorm2d if instance_dimensionality == '2D' else torch.nn.BatchNorm1d
        elif normalization=='Identity':
            Norm = torch.nn.Identity # class, not object, we instiate per layer with actual in_channels

        Conv = torch.nn.Conv2d if instance_dimensionality == '2D' else torch.nn.Conv1d

        self.layers = []
        for n in range(len(out_channels)):
            self.layers.append(
                BilinearConvLayer(input_channels=input_channels if n==0 else out_channels[n-1],
                                  output_channels=out_channels[n],
                                  bilin_channels=bilin_channels[n],
                                  padding_mode=padding_mode,
                                  nonlinearity=nonlinearity,
                                  norm=Norm(out_channels[n]+bilin_channels[n]),
                                  Conv=Conv)
            )
            # final conv layer without nonlinearity:
        self.layers.append(Conv(max(bilin_channels[-1], out_channels[-1]),
                                out_channels[-1],
                                kernel_size=1))
        self.layers = torch.nn.ModuleList(self.layers)        

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        return x
