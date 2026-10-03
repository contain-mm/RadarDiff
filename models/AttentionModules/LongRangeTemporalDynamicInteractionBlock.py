import math
from functools import partial
from collections import namedtuple
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from utill.Tools import Tools




class LongRangeTemporalDynamicInteractionBlock(nn.Module):

    def __init__(self, d_model, kernel_size=21, attn_shortcut=True):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)  # 1x1 conv
        self.activation = nn.GELU()  # GELU
        self.spatial_gating_unit = AdaptiveChannelInteractionBlock(d_model,kernel_size)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)  # 1x1 conv
        self.attn_shortcut = attn_shortcut

    def forward(self, x):
        if self.attn_shortcut:
            shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        if self.attn_shortcut:
            x = x + shortcut
        return x


class AdaptiveChannelInteractionBlock(nn.Module):

    def __init__(self, dim, kernel_size,dilation=3):
        super().__init__()
        d_k = 2 * dilation - 1
        d_p = (d_k - 1) // 2
        dd_k = kernel_size // dilation + ((kernel_size // dilation) % 2 - 1)
        dd_p = (dilation * (dd_k - 1) // 2)
        self.conv0 = nn.Conv2d(dim, dim, d_k, padding=d_p, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, dd_k, stride=1, padding=dd_p, groups=dim, dilation=dilation)
        self.conv1 = nn.Conv2d(dim, dim, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        kernel_sz = int(abs((math.log(dim, 2) + 1) / 2))
        kernel_sz = kernel_sz if kernel_sz % 2 else kernel_sz + 1
        self.conv = nn.Conv1d(1,1,kernel_size=kernel_sz,padding=(kernel_sz-1)//2,bias=False)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // 1, bias=False),  # reduction
            nn.ReLU(True),
            nn.Linear(dim // 1, dim, bias=False),  # expansion
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)  # depth-wise conv
        attn = self.conv_spatial(attn)  # depth-wise dilation convolution
        f_x = self.conv1(attn)  # 1x1 conv

        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y_local = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1).expand_as(x)
        y_global = self.fc(y.view(b, c)).view(b, c, 1, 1).expand_as(x)
        y = y_local + y_global
        se_atten = self.sigmoid(y)
        out = se_atten*f_x*u
        return out  
