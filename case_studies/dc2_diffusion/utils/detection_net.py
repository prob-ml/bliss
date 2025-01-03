import math

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from bliss.encoder.convnet_layers import C3, ConvBlock

def exists(x):
    return x is not None

def divisible_by(numer, denom):
    return (numer % denom) == 0


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered


class Block(nn.Module):
    def __init__(self, dim, dim_out, dropout=0.0, groups=32):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, (1, 1), padding=0)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return self.dropout(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim, dropout=0.0, groups=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out)
        )

        self.block1 = Block(dim, dim_out, dropout=dropout, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, (1, 1)) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        time_emb = self.mlp(time_emb)
        time_emb = rearrange(time_emb, "b c -> b c 1 1")
        h = self.block1(x)
        h = h + time_emb
        h = self.block2(h)
        return h + self.res_conv(x)


class DetectionNet(nn.Module):
    def __init__(self, xt_in_ch, xt_out_ch, extracted_feats_ch, use_self_cond=False):
        super().__init__()

        self.xt_in_ch = xt_in_ch
        self.xt_out_ch = xt_out_ch
        self.extracted_feats_ch = extracted_feats_ch
        self.dim = self.xt_out_ch + self.extracted_feats_ch
        self.use_self_cond = use_self_cond
        if self.use_self_cond:
            self.dim += self.xt_in_ch
        self.time_emb_dim = 4 * self.dim
        self.final_out_ch = xt_in_ch

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(self.dim),
            nn.Linear(self.dim, self.time_emb_dim),
            nn.SiLU(),
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
        )
        
        if self.xt_out_ch > 8:
            assert self.xt_out_ch % 8 == 0
        self.xt_preprocess = Block(self.xt_in_ch, self.xt_out_ch, 
                                   dropout=0, 
                                   groups=1 if self.xt_out_ch <= 8
                                          else self.xt_out_ch // 8)
        self.xt_process = ResnetBlock(
            self.xt_out_ch, self.xt_out_ch, 
            time_emb_dim=self.time_emb_dim, dropout=0,
            groups=1 if self.xt_out_ch <= 8
                   else self.xt_out_ch // 8
        )

        assert self.extracted_feats_ch % 8 == 0
        self.feats_preprocess = Block(self.extracted_feats_ch, self.extracted_feats_ch,
                                      dropout=0,
                                      groups=self.extracted_feats_ch // 8)
        self.feats_process = ResnetBlock(
            self.extracted_feats_ch,
            self.extracted_feats_ch,
            time_emb_dim=self.time_emb_dim,
            dropout=0,
            groups=self.extracted_feats_ch // 8
        )

        if self.dim % 8 == 0:
            xt_feats_groups = self.dim // 8
        elif self.dim % 4 == 0:
            xt_feats_groups = self.dim // 4
        else:
            raise ValueError("dim is not divisible by 8 or 4")
        self.xt_feats_process = nn.ModuleList(
            [
                ResnetBlock(self.dim, self.dim, 
                            time_emb_dim=self.time_emb_dim, 
                            dropout=0,
                            groups=xt_feats_groups)
                for _ in range(4)
            ]
        )
        self.final_process = nn.Sequential(
            Block(self.dim, self.dim, dropout=0.0, groups=xt_feats_groups),
            nn.Conv2d(self.dim, self.final_out_ch, (1, 1), padding=0)
        )

    def forward(self, x, time, extracted_feats, self_cond=None):
        if self.use_self_cond and self_cond is None:
            self_cond = torch.zeros_like(x)
        if not self.use_self_cond:
            assert self_cond is None
        t = self.time_mlp(time)
        x = self.xt_preprocess(x)
        x = self.xt_process(x, t)
        extracted_feats = self.feats_preprocess(extracted_feats)
        extracted_feats = self.feats_process(extracted_feats, t)
        x = torch.cat([x, extracted_feats] 
                      if not self.use_self_cond 
                      else [x, extracted_feats, self_cond], 
                      dim=1)
        for layer in self.xt_feats_process:
            x = layer(x, t)
        return self.final_process(x)
    

class TimeScaleShiftBlock(nn.Module):
    def __init__(self, time_emb_dim, input_dim):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.input_dim = input_dim
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.time_emb_dim, self.input_dim * 2)
        )
    
    def forward(self, x, time):
        scale_shift = rearrange(self.time_mlp(time), "b c -> b c 1 1")
        scale, shift = scale_shift.chunk(2, dim=1)
        return x * (scale + 1) + shift
    

class SimpleDetectionNet(nn.Module):
    def __init__(self, xt_in_ch, xt_out_ch, extracted_feats_ch, final_out_ch=None):
        super().__init__()

        self.xt_in_ch = xt_in_ch
        self.xt_out_ch = xt_out_ch
        self.extracted_feats_ch = extracted_feats_ch
        self.dim = self.xt_out_ch + self.extracted_feats_ch
        self.xt_time_emb_dim = 4 * self.xt_out_ch
        self.feats_time_emb_dim = 4 * self.extracted_feats_ch
        if final_out_ch:
            self.final_out_ch = final_out_ch
        else:
            self.final_out_ch = xt_in_ch

        self.xt_time_mlp = nn.Sequential(
            SinusoidalPosEmb(self.xt_out_ch),
            nn.Linear(self.xt_out_ch, self.xt_time_emb_dim),
            nn.SiLU(),
            nn.Linear(self.xt_time_emb_dim, self.xt_time_emb_dim),
        )
        self.feats_time_mlp = nn.Sequential(
            SinusoidalPosEmb(self.extracted_feats_ch),
            nn.Linear(self.extracted_feats_ch, self.feats_time_emb_dim),
            nn.SiLU(),
            nn.Linear(self.feats_time_emb_dim, self.feats_time_emb_dim),
        )
        
        if self.xt_out_ch > 8:
            assert self.xt_out_ch % 8 == 0
        self.xt_preprocess = nn.Sequential(
            ConvBlock(self.xt_in_ch, self.xt_out_ch, kernel_size=1, gn=True),
            C3(self.xt_out_ch, self.xt_out_ch, n=3, spatial=False, gn=True),
            ConvBlock(self.xt_out_ch, self.xt_out_ch, kernel_size=1, gn=True),
        )
        self.xt_scale_shift = TimeScaleShiftBlock(self.xt_time_emb_dim, self.xt_out_ch)

        # assert self.extracted_feats_ch % 8 == 0
        # self.feats_preprocess = Block(self.extracted_feats_ch, self.extracted_feats_ch,
        #                               dropout=0,
        #                               groups=self.extracted_feats_ch // 8)
        self.feats_scale_shift = TimeScaleShiftBlock(self.feats_time_emb_dim, self.extracted_feats_ch)
        
        self.final_process = nn.Sequential(
            ConvBlock(self.dim, self.dim, kernel_size=1, gn=True),
            C3(self.dim, self.dim, n=3, spatial=False, gn=True),
            ConvBlock(self.dim, self.dim, kernel_size=1, gn=True),
            C3(self.dim, self.dim, n=3, spatial=False, gn=True),
            ConvBlock(self.dim, self.dim, kernel_size=1, gn=True),
            nn.Conv2d(self.dim, self.final_out_ch, (1, 1), padding=0)
        )

    def forward(self, x, time, extracted_feats, self_cond=None):
        assert self_cond is None
        xt_t = self.xt_time_mlp(time)
        feats_t = self.feats_time_mlp(time)
        x = self.xt_preprocess(x)
        x = self.xt_scale_shift(x, xt_t)
        # extracted_feats = self.feats_preprocess(extracted_feats)
        extracted_feats = self.feats_scale_shift(extracted_feats, feats_t)
        x = torch.cat([x, extracted_feats], dim=1)
        return self.final_process(x)
