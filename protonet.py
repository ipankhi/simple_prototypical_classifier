import torch.nn as nn
import numpy as np
def conv_block (in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=False),
        nn.MaxPool2d(2)
    )

class ProtoNet(nn.Module):
    def __init__(self, x_dim = 1, hid_dim = 64, z_dim = 64):
        super(ProtoNet, self).__init__()
        self.encoder=nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim,hid_dim),
            conv_block(hid_dim, z_dim)
        )
    def forward(self, x):
        x = x.transpose(1, 3) 
        print(x.shape)
        x = self.encoder(x)
        return x.view(x.size(0), -1)