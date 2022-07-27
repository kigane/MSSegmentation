from functools import partial

import torch
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

import torch.nn.functional as F
from torchsnooper import snoop

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# helper classes

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# MBConv

class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate = 0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(dim, hidden_dim, bias = False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias = False),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1 1')
        )

    def forward(self, x):
        return x * self.gate(x)


class MBConvResidual(nn.Module):
    def __init__(self, fn, dropout = 0.):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x

class Dropsample(nn.Module):
    def __init__(self, prob = 0):
        super().__init__()
        self.prob = prob
  
    def forward(self, x):
        device = x.device

        if self.prob == 0. or (not self.training):
            return x

        keep_mask = torch.FloatTensor((x.shape[0], 1, 1, 1), device = device).uniform_() > self.prob
        return x * keep_mask / (1 - self.prob)

def MBConv(
    dim_in,
    dim_out,
    *,
    downsample,
    expansion_rate = 4,
    shrinkage_rate = 0.25,
    dropout = 0.
):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1

    net = nn.Sequential(
        nn.Conv2d(dim_in, hidden_dim, 1),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        nn.Conv2d(hidden_dim, hidden_dim, 3, stride = stride, padding = 1, groups = hidden_dim),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        SqueezeExcitation(hidden_dim, shrinkage_rate = shrinkage_rate),
        nn.Conv2d(hidden_dim, dim_out, 1),
        nn.BatchNorm2d(dim_out)
    )

    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout = dropout)

    return net

# attention related classes

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 32,
        dropout = 0.,
        window_size = 7
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias = False),
            nn.Dropout(dropout)
        )

        # relative positional bias

        self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)

        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos))
        grid = rearrange(grid, 'c i j -> (i j) c')
        rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim = -1)

        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent = False)

    def forward(self, x):
        batch, height, width, window_height, window_width, _, device, h = *x.shape, x.device, self.heads

        # flatten

        x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')

        # project for queries, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # split heads

        q, k, v = map(lambda t: rearrange(t, 'b n (h d ) -> b h n d', h = h), (q, k, v))

        # scale

        q = q * self.scale

        # sim

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # add positional bias

        bias = self.rel_pos_bias(self.rel_pos_indices)
        sim = sim + rearrange(bias, 'i j h -> h i j')

        # attention

        attn = self.attend(sim)

        # aggregate

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads

        out = rearrange(out, 'b h (w1 w2) d -> b w1 w2 (h d)', w1 = window_height, w2 = window_width)

        # combine heads out

        out = self.to_out(out)
        return rearrange(out, '(b x y) ... -> b x y ...', x = height, y = width)

class SkipAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 32,
        dropout = 0.,
        window_size = 7
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'
        self.dim = dim
        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5

        self.to_kv = nn.Linear(dim, dim * 2, bias = False)
        # self.to_q = nn.Linear(dim, dim, bias = False)

        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias = False),
            nn.Dropout(dropout)
        )

        # relative positional bias

        self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)

        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos))
        grid = rearrange(grid, 'c i j -> (i j) c')
        rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim = -1)

        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent = False)

    def forward(self, x, skip):
        batch, height, width, window_height, window_width, _, device, h = *x.shape, x.device, self.heads

        # flatten
        x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')
        skip = rearrange(skip, 'b x y w1 w2 d -> (b x y) (w1 w2) d')

        # project for queries, keys, values
        k, v = self.to_kv(skip).chunk(2, dim = -1)
        # q = self.to_q(x)
        q = x

        # norm
        q, k, v = map(lambda t: F.layer_norm(t, (self.dim,)), (q, k, v))

        # split heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d ) -> b h n d', h = h), (q, k, v))

        # scale
        q = q * self.scale

        # sim
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # add positional bias
        bias = self.rel_pos_bias(self.rel_pos_indices)
        sim = sim + rearrange(bias, 'i j h -> h i j')

        # attention
        attn = self.attend(sim)

        # aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads
        out = rearrange(out, 'b h (w1 w2) d -> b w1 w2 (h d)', w1 = window_height, w2 = window_width)

        # combine heads out
        out = self.to_out(out)
        return rearrange(out, '(b x y) ... -> b x y ...', x = height, y = width)

class MaxViTBlock(nn.Module):
    def __init__(
        self,
        chs_in, 
        chs_out,
        dim_head=32,
        downsample=True,
        dropout=0.,
        window_size=7,
        mbconv_expansion_rate=4, 
        mbconv_shrinkage_rate=0.25
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
                MBConv(
                    chs_in,
                    chs_out,
                    downsample = downsample,
                    expansion_rate = mbconv_expansion_rate,
                    shrinkage_rate = mbconv_shrinkage_rate
                ),
                Rearrange('b d (x w1) (y w2) -> b x y w1 w2 d', w1 = window_size, w2 = window_size),  # block-like attention
                PreNormResidual(chs_out, Attention(dim = chs_out, dim_head = dim_head, dropout = dropout, window_size = window_size)),
                PreNormResidual(chs_out, FeedForward(dim = chs_out, dropout = dropout)),
                Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)'),

                Rearrange('b d (w1 x) (w2 y) -> b x y w1 w2 d', w1 = window_size, w2 = window_size),  # grid-like attention
                PreNormResidual(chs_out, Attention(dim = chs_out, dim_head = dim_head, dropout = dropout, window_size = window_size)),
                PreNormResidual(chs_out, FeedForward(dim = chs_out, dropout = dropout)),
                Rearrange('b x y w1 w2 d -> b d (w1 x) (w2 y)'),
            )
    
    def forward(self, x):
        return self.block(x)


class MaxViTUpBlock(nn.Module):
    def __init__(
        self,
        chs_in, 
        chs_out,
        dim_head=32,
        dropout=0.,
        window_size=7,
        mbconv_expansion_rate=4, 
        mbconv_shrinkage_rate=0.25
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(chs_in, chs_out, kernel_size=3, padding=1)
        self.block = nn.Sequential(
                MBConv(
                    chs_in,
                    chs_out,
                    downsample = False,
                    expansion_rate = mbconv_expansion_rate,
                    shrinkage_rate = mbconv_shrinkage_rate
                ),
                Rearrange('b d (x w1) (y w2) -> b x y w1 w2 d', w1 = window_size, w2 = window_size),  # block-like attention
                PreNormResidual(chs_out, Attention(dim = chs_out, dim_head = dim_head, dropout = dropout, window_size = window_size)),
                PreNormResidual(chs_out, FeedForward(dim = chs_out, dropout = dropout)),
                Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)'),

                Rearrange('b d (w1 x) (w2 y) -> b x y w1 w2 d', w1 = window_size, w2 = window_size),  # grid-like attention
                PreNormResidual(chs_out, Attention(dim = chs_out, dim_head = dim_head, dropout = dropout, window_size = window_size)),
                PreNormResidual(chs_out, FeedForward(dim = chs_out, dropout = dropout)),
                Rearrange('b x y w1 w2 d -> b d (w1 x) (w2 y)'),
            )
    
    @snoop()
    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=(2, 2), mode="bilinear", align_corners=True)
        x = self.conv(x)
        x = torch.cat([skip, x], dim=1)
        return self.block(x)

class MaxViTSkipUpBlock(nn.Module):
    def __init__(
        self,
        chs_in, 
        chs_out,
        dim_head=32,
        dropout=0.,
        window_size=7,
        mbconv_expansion_rate=4, 
        mbconv_shrinkage_rate=0.25
    ) -> None:
        super().__init__()
        self.dim = chs_out
        self.ws = window_size
        self.dh = dim_head

        self.mbconv = MBConv(
                    chs_in,
                    chs_out,
                    downsample = False,
                    expansion_rate = mbconv_expansion_rate,
                    shrinkage_rate = mbconv_shrinkage_rate
                )

        self.img_to_block = Rearrange('b d (x w1) (y w2) -> b x y w1 w2 d', w1 = window_size, w2 = window_size)
        self.skip_block_attn = SkipAttention(dim = chs_out, dim_head = dim_head, dropout = dropout, window_size = window_size)
        # self.ln2 = nn.LayerNorm(chs_out)
        self.block_ffn = FeedForward(dim = chs_out, dropout = dropout)
        # self.ln3 = nn.LayerNorm(chs_out)
        self.block_to_img = Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)')

        self.img_to_grid = Rearrange('b d (w1 x) (w2 y) -> b x y w1 w2 d', w1 = window_size, w2 = window_size)
        self.skip_grid_attn = SkipAttention(dim = chs_out, dim_head = dim_head, dropout = dropout, window_size = window_size)
        # self.ln4 = nn.LayerNorm(chs_out)
        self.grid_ffn = FeedForward(dim = chs_out, dropout = dropout)
        # self.ln5 = nn.LayerNorm(chs_out)
        self.grid_to_img = Rearrange('b x y w1 w2 d -> b d (w1 x) (w2 y)')

        self.block = nn.Sequential(
                MBConv(
                    chs_out,
                    chs_out,
                    downsample = False,
                    expansion_rate = mbconv_expansion_rate,
                    shrinkage_rate = mbconv_shrinkage_rate
                ),
                Rearrange('b d (x w1) (y w2) -> b x y w1 w2 d', w1 = window_size, w2 = window_size),  # block-like attention
                PreNormResidual(chs_out, Attention(dim = chs_out, dim_head = dim_head, dropout = dropout, window_size = window_size)),
                PreNormResidual(chs_out, FeedForward(dim = chs_out, dropout = dropout)),
                Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)'),

                Rearrange('b d (w1 x) (w2 y) -> b x y w1 w2 d', w1 = window_size, w2 = window_size),  # grid-like attention
                PreNormResidual(chs_out, Attention(dim = chs_out, dim_head = dim_head, dropout = dropout, window_size = window_size)),
                PreNormResidual(chs_out, FeedForward(dim = chs_out, dropout = dropout)),
                Rearrange('b x y w1 w2 d -> b d (w1 x) (w2 y)'),
            )

    @snoop()
    def forward(self, x, skip):
        ln = partial(F.layer_norm, normalized_shape=(self.dim,))
        # x [1, 256, 14, 14] -> [1, 256, 28, 28]
        x = F.interpolate(x, scale_factor=(2, 2), mode="bilinear", align_corners=True)
        # x [1, 256, 28, 28] -> [1, 128, 28, 28]
        x = self.mbconv(x)
        # skip [1, 128, 28, 28]

        x = self.img_to_block(x)
        skip = self.img_to_block(skip)
        short_cut = x + skip # x_up + x_{lv,gv}

        x = ln(x)
        skip = ln(skip)

        attn_ret_1 = self.skip_block_attn(x, skip)
        attn_ret_1 = attn_ret_1 + short_cut # attention residual
        attn_ret_1 = ln(attn_ret_1)
        attn_ret = self.block_ffn(attn_ret_1)
        attn_ret = attn_ret + attn_ret_1 # ffn residual

        x = ln(x)
        x = self.block_to_img(attn_ret)
        skip = self.block_to_img(skip)
        x = self.img_to_grid(x)
        skip = self.img_to_grid(skip)
        short_cut = x + skip

        attn_ret_2 = self.skip_grid_attn(x, skip)
        attn_ret_2 = attn_ret_2 + short_cut # attention residual
        attn_ret_2 = ln(attn_ret_2)
        attn_ret = self.grid_ffn(attn_ret_2)
        attn_ret = attn_ret + attn_ret_2 # ffn residual

        x = self.grid_to_img(attn_ret)

        return self.block(x)


class MaxViTUnet(nn.Module):
    def __init__(
        self,
        channels = 3,                   # 输入图片的通道数
        *,
        features=[16, 32, 64, 128, 256],# 特征通道数
        dim_head = 16,                  # attention head的维度
        window_size = 7,                # G, P
        mbconv_expansion_rate = 4,      # MBconv中conv1x1通道扩大倍数
        mbconv_shrinkage_rate = 0.25,   # SE中通道缩小倍数
        dropout = 0.1,
    ):
        super().__init__()

        self.conv_stem = nn.Sequential(
            nn.Conv2d(channels, features[0], 3, padding = 1),
            nn.Conv2d(features[0], features[0], 3, padding = 1)
        )

        self.down1 = MaxViTBlock(
            features[0], features[1], dim_head=dim_head,
            dropout=dropout, window_size=window_size,
            mbconv_expansion_rate=mbconv_expansion_rate,
            mbconv_shrinkage_rate=mbconv_shrinkage_rate)

        self.down2 = MaxViTBlock(
            features[1], features[2], dim_head=dim_head,
            dropout=dropout, window_size=window_size,
            mbconv_expansion_rate=mbconv_expansion_rate,
            mbconv_shrinkage_rate=mbconv_shrinkage_rate)

        self.down3 = MaxViTBlock(
            features[2], features[3], dim_head=dim_head,
            dropout=dropout, window_size=window_size,
            mbconv_expansion_rate=mbconv_expansion_rate,
            mbconv_shrinkage_rate=mbconv_shrinkage_rate)
        
        self.down4 = MaxViTBlock(
            features[3], features[4], dim_head=dim_head,
            dropout=dropout, window_size=window_size,
            mbconv_expansion_rate=mbconv_expansion_rate,
            mbconv_shrinkage_rate=mbconv_shrinkage_rate)
        
        self.bottleneck = MaxViTBlock(
            features[4], features[4], dim_head=dim_head,
            downsample=False, dropout=dropout, 
            window_size=window_size,
            mbconv_expansion_rate=mbconv_expansion_rate,
            mbconv_shrinkage_rate=mbconv_shrinkage_rate)
        
        self.up4 = MaxViTUpBlock(
            features[4], features[3], dim_head=dim_head,
            dropout=dropout, window_size=window_size,
            mbconv_expansion_rate=mbconv_expansion_rate,
            mbconv_shrinkage_rate=mbconv_shrinkage_rate)
        
        self.up3 = MaxViTUpBlock(
            features[3], features[2], dim_head=dim_head,
            dropout=dropout, window_size=window_size,
            mbconv_expansion_rate=mbconv_expansion_rate,
            mbconv_shrinkage_rate=mbconv_shrinkage_rate)
        
        self.up2 = MaxViTUpBlock(
            features[2], features[1], dim_head=dim_head,
            dropout=dropout, window_size=window_size,
            mbconv_expansion_rate=mbconv_expansion_rate,
            mbconv_shrinkage_rate=mbconv_shrinkage_rate)
        
        self.up1 = MaxViTUpBlock(
            features[1], features[0], dim_head=dim_head,
            dropout=dropout, window_size=window_size,
            mbconv_expansion_rate=mbconv_expansion_rate,
            mbconv_shrinkage_rate=mbconv_shrinkage_rate)
         
        self.final_conv = nn.Conv2d(features[0], 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv_stem(x)  # [1,16,224,224]
        x2 = self.down1(x1)     # [1,32,112,112]
        x3 = self.down2(x2)     # [1,64,56,56]
        x4 = self.down3(x3)     # [1,128,28,28]
        x5 = self.down4(x4)     # [1,256,14,14]
        x = self.bottleneck(x5) # [1,256,14,14]
        x6 = self.up4(x, x4)    # [1,128,28,28]
        x7 = self.up3(x6, x3)   # [1,64,56,56]
        x8 = self.up2(x7, x2)   # [1,32,112,112]
        x9 = self.up1(x8, x1)   # [1,16,224,224]
        x = self.final_conv(x9) # [1,1,224,224]
        return self.sigmoid(x)

class MaxViTSkipUnet(nn.Module):
    def __init__(
        self,
        channels = 3,                   # 输入图片的通道数
        *,
        features=[16, 32, 64, 128, 256],# 特征通道数
        dim_head = 16,                  # attention head的维度
        window_size = 7,                # G, P
        mbconv_expansion_rate = 4,      # MBconv中conv1x1通道扩大倍数
        mbconv_shrinkage_rate = 0.25,   # SE中通道缩小倍数
        dropout = 0.1,
    ):
        super().__init__()

        self.conv_stem = nn.Sequential(
            nn.Conv2d(channels, features[0], 3, padding = 1),
            nn.Conv2d(features[0], features[0], 3, padding = 1)
        )

        self.down1 = MaxViTBlock(
            features[0], features[1], dim_head=dim_head,
            dropout=dropout, window_size=window_size,
            mbconv_expansion_rate=mbconv_expansion_rate,
            mbconv_shrinkage_rate=mbconv_shrinkage_rate)

        self.down2 = MaxViTBlock(
            features[1], features[2], dim_head=dim_head,
            dropout=dropout, window_size=window_size,
            mbconv_expansion_rate=mbconv_expansion_rate,
            mbconv_shrinkage_rate=mbconv_shrinkage_rate)

        self.down3 = MaxViTBlock(
            features[2], features[3], dim_head=dim_head,
            dropout=dropout, window_size=window_size,
            mbconv_expansion_rate=mbconv_expansion_rate,
            mbconv_shrinkage_rate=mbconv_shrinkage_rate)
        
        self.down4 = MaxViTBlock(
            features[3], features[4], dim_head=dim_head,
            dropout=dropout, window_size=window_size,
            mbconv_expansion_rate=mbconv_expansion_rate,
            mbconv_shrinkage_rate=mbconv_shrinkage_rate)
        
        self.bottleneck = MaxViTBlock(
            features[4], features[4], dim_head=dim_head,
            downsample=False, dropout=dropout, 
            window_size=window_size,
            mbconv_expansion_rate=mbconv_expansion_rate,
            mbconv_shrinkage_rate=mbconv_shrinkage_rate)
        
        self.up4 = MaxViTSkipUpBlock(
            features[4], features[3], dim_head=dim_head,
            dropout=dropout, window_size=window_size,
            mbconv_expansion_rate=mbconv_expansion_rate,
            mbconv_shrinkage_rate=mbconv_shrinkage_rate)
        
        self.up3 = MaxViTSkipUpBlock(
            features[3], features[2], dim_head=dim_head,
            dropout=dropout, window_size=window_size,
            mbconv_expansion_rate=mbconv_expansion_rate,
            mbconv_shrinkage_rate=mbconv_shrinkage_rate)
        
        self.up2 = MaxViTSkipUpBlock(
            features[2], features[1], dim_head=dim_head,
            dropout=dropout, window_size=window_size,
            mbconv_expansion_rate=mbconv_expansion_rate,
            mbconv_shrinkage_rate=mbconv_shrinkage_rate)
        
        self.up1 = MaxViTSkipUpBlock(
            features[1], features[0], dim_head=dim_head,
            dropout=dropout, window_size=window_size,
            mbconv_expansion_rate=mbconv_expansion_rate,
            mbconv_shrinkage_rate=mbconv_shrinkage_rate)
         
        self.final_conv = nn.Conv2d(features[0], 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv_stem(x)  # [1,16,224,224]
        x2 = self.down1(x1)     # [1,32,112,112]
        x3 = self.down2(x2)     # [1,64,56,56]
        x4 = self.down3(x3)     # [1,128,28,28]
        x5 = self.down4(x4)     # [1,256,14,14]
        x = self.bottleneck(x5) # [1,256,14,14]
        x6 = self.up4(x, x4)    # [1,128,28,28]
        x7 = self.up3(x6, x3)   # [1,64,56,56]
        x8 = self.up2(x7, x2)   # [1,32,112,112]
        x9 = self.up1(x8, x1)   # [1,16,224,224]
        x = self.final_conv(x9) # [1,1,224,224]
        return self.sigmoid(x)

if __name__ == '__main__':
    # b = MaxViTBlock(16, 32)
    # x = torch.randn(1, 16, 56, 56)
    # print(b(x).shape) # (1, 32, 28, 28)
    x = torch.randn(1, 1, 224, 224)
    model = MaxViTSkipUnet(channels=1)
    y = model(x)
    print(y.shape)
