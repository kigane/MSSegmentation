from functools import partial

import torch
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

import torch.nn.functional as F

from models.MaxViT import MaxViTBlock, MBConvResidual
from models.utils import get_activation


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        act='relu'
    ):
        super(DoubleConv, self).__init__()
        mlist = []
        mlist.append(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, 1, kernel_size // 2, bias=False
            )
        )
        mlist.append(nn.BatchNorm2d(out_channels))
        mlist.append(get_activation(act))
        mlist.append(
            nn.Conv2d(
                out_channels, out_channels, kernel_size, 1, kernel_size // 2, bias=False
            )
        )
        mlist.append(nn.BatchNorm2d(out_channels))
        mlist.append(get_activation(act))
        self.dconv = nn.Sequential(*mlist)

    def forward(self, x):
        return self.dconv(x)


class UpConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        act='relu'
    ):
        super().__init__()
        self.down_ch = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.dconv = DoubleConv(in_channels, out_channels, act=act)
    
    def forward(self, x, skip):
        x = self.down_ch(F.interpolate(x, scale_factor=(2, 2), mode='bilinear', align_corners=True))
        x = self.dconv(torch.cat([x, skip], dim=1))
        return x

class HybridMVUnet(nn.Module):
    def __init__(
        self,
        channels = 3,                   # 输入图片的通道数
        out_channels = 1,
        *,
        features = [16, 32, 64, 128, 256],# 特征通道数
        dim_head = 16,                  # attention head的维度
        window_size = 7,                # G, P
        mbconv_expansion_rate = 4,      # MBconv中conv1x1通道扩大倍数
        mbconv_shrinkage_rate = 0.25,   # SE中通道缩小倍数
        dropout = 0.1,
    ) -> None:
        super().__init__()
        
        self.encoder1 = DoubleConv(channels, features[0])       # 224
        self.encoder2 = DoubleConv(features[0], features[1])    # 112
        self.encoder3 = DoubleConv(features[1], features[2])    # 56
        
        self.pool = nn.MaxPool2d(2, 2)

        self.max_encoder = MaxViTBlock(features[2], features[3], downsample=False, 
                    dim_head=16, dropout=dropout, window_size=window_size,
                    mbconv_expansion_rate=mbconv_expansion_rate,
                    mbconv_shrinkage_rate=mbconv_shrinkage_rate) # 28

        self.max_bottlenect = MaxViTBlock(features[3], features[4], downsample=False, 
                    dim_head=16, dropout=dropout, window_size=window_size,
                    mbconv_expansion_rate=mbconv_expansion_rate,
                    mbconv_shrinkage_rate=mbconv_shrinkage_rate) # 14    

        self.down_ch = nn.Conv2d(features[4], features[3], 3, 1, 1, bias=False)

        self.max_decoder = MaxViTBlock(features[4], features[3], downsample=False, 
                    dim_head=16, dropout=dropout, window_size=window_size,
                    mbconv_expansion_rate=mbconv_expansion_rate,
                    mbconv_shrinkage_rate=mbconv_shrinkage_rate) # 28

        self.decoder3 = UpConv(features[3], features[2])     # 56  
        self.decoder2 = UpConv(features[2], features[1])     # 112
        self.decoder1 = UpConv(features[1], features[0])     # 224

        self.final_conv = nn.Conv2d(features[0], out_channels, 1, 1)

    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.encoder2(self.pool(x1)) 
        x3 = self.encoder3(self.pool(x2)) 
        x4 = self.max_encoder(self.pool(x3))
        x5 = self.max_bottlenect(self.pool(x4))
        x5 = F.interpolate(x5, scale_factor=(2, 2), mode='bilinear', align_corners=True)
        x5 = self.down_ch(x5)
        x6 = self.max_decoder(torch.cat([x5, x4], dim=1))
        x7 = self.decoder3(x6, x3)
        x8 = self.decoder2(x7, x2)
        x9 = self.decoder1(x8, x1)
        x10 = self.final_conv(x9)
        return torch.sigmoid(x10)

if __name__ == '__main__':
    model = HybridMVUnet(1, 1)
    x = torch.randn(1, 1, 224, 224)
    y = model(x)
    print(y.shape)