import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from functools import partial
from typing import List
import math

from case_studies.dc2_new_diffusion.utils.convnet_layers import C3, ConvBlock

class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        half_d_model = d_model // 2
        log_denominator = -math.log(10000) / (half_d_model - 1)
        denominator_ = torch.exp(torch.arange(half_d_model) * log_denominator)
        self.register_buffer("denominator", denominator_)

    def forward(self, time):
        """
        :param time: shape=(B, )
        :return: Positional Encoding shape=(B, d_model)
        """
        argument = time[:, None] * self.denominator[None, :]  # (B, half_d_model)
        return torch.cat([argument.sin(), argument.cos()], dim=-1)  # (B, d_model)


# class RMSNorm(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

#     def forward(self, x):
#         return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)


class Block(nn.Module):
    def __init__(self, dim, dim_out, spatial=True):
        """
        Input shape=(B, dim, H, W)
        Output shape=(B, dim_out, H, W)

        :param dim: input channel
        :param dim_out: output channel
        :param groups: number of groups for Group normalization.
        """
        super().__init__()
        if spatial:
            self.proj = nn.Conv2d(dim, dim_out, kernel_size=(3, 3), padding=1)
        else:
            self.proj = nn.Conv2d(dim, dim_out, kernel_size=(1, 1), padding=0)
        assert dim_out % 8 == 0
        self.norm = nn.GroupNorm(num_groups=dim_out // 8, num_channels=dim_out)
        self.activation = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (1 + scale) + shift
        return self.activation(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, spatial=True):
        """
        In abstract, it is composed of two Convolutional layer with residual connection,
        with information of time encoding is passed to first Convolutional layer.

        Input shape=(B, dim, H, W)
        Output shape=(B, dim_out, H, W)

        :param dim: input channel
        :param dim_out: output channel
        :param time_emb_dim: Embedding dimension for time.
        :param group: number of groups for Group normalization.
        """
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) if time_emb_dim is not None else None
        self.block1 = Block(dim, dim_out, spatial=spatial)
        self.block2 = Block(dim_out, dim_out, spatial=spatial)
        self.residual_conv = nn.Conv2d(dim, dim_out, kernel_size=(1, 1)) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        """

        :param x: (B, dim, H, W)
        :param time_emb: (B, time_emb_dim)
        :return: (B, dim_out, H, W)
        """
        scale_shift = None
        if time_emb is not None:
            scale_shift = self.mlp(time_emb)[..., None, None]  # (B, dim_out*2, 1, 1)
            scale_shift = scale_shift.chunk(2, dim=1)  # len 2 with each element shape (B, dim_out, 1, 1)
        hidden = self.block1(x, scale_shift)
        hidden = self.block2(hidden)
        return hidden + self.residual_conv(x)


class Attention(nn.Module):
    def __init__(self, dim, head=4, dim_head=32):
        super().__init__()
        self.head = head
        hidden_dim = head * dim_head

        self.scale = dim_head ** (-0.5)  # 1 / sqrt(d_k)
        assert dim % 8 == 0
        self.norm = nn.GroupNorm(num_groups=dim // 8, num_channels=dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, kernel_size=(1, 1), bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, kernel_size=(1, 1))

    def forward(self, x):
        b, c, i, j = x.shape
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        # h=self.head, f=dim_head, i=height, j=width.
        # You can think (i*j) as sequence length where f is d_k in <Attention is all you need>
        q, k, v = map(lambda t: rearrange(t, "b (h f) i j -> b h (i j) f", h=self.head), qkv)

        """
        q, k, v shape: (batch, # of head, seq_length, d_k)  seq_length = height * width
        similarity shape: (batch, # of head, seq_length, seq_length)
        attention_score shape: (batch, # of head, seq_length, seq_length)
        attention shape: (batch, # of head, seq_length, d_k)
        out shape: (batch, hidden_dim, height, width)
        return shape: (batch, dim, height, width)
        """
        # n, m is likewise sequence length.
        similarity = torch.einsum("b h n f, b h m f -> b h n m", q, k)  # Q(K^T)
        attention_score = torch.softmax(similarity * self.scale, dim=-1)  # softmax(Q(K^T) / sqrt(d_k))
        attention = torch.einsum("b h n m, b h m f -> b h n f", attention_score, v)
        # attention(Q, K, V) = [softmax(Q(K^T) / sqrt(d_k))]V -> Scaled Dot-Product Attention

        out = rearrange(attention, "b h (i j) f -> b (h f) i j", i=i, j=j)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, head=4, dim_head=32):
        super().__init__()
        self.head = head
        hidden_dim = head * dim_head

        self.scale = dim_head ** (-0.5)
        assert dim % 8 == 0
        self.norm = nn.GroupNorm(num_groups=dim // 8, num_channels=dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, kernel_size=(1, 1), bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, kernel_size=(1, 1)), 
            nn.GroupNorm(num_groups=dim // 8, num_channels=dim)
        )

    def forward(self, x):
        b, c, i, j = x.shape
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        # h=self.head, f=dim_head, i=height, j=width.
        # You can think (i*j) as sequence length where f is d_k in <Attention is all you need>
        q, k, v = map(lambda t: rearrange(t, "b (h f) i j -> b h f (i j)", h=self.head), qkv)

        q = q.softmax(dim=-2) * self.scale
        k = k.softmax(dim=-1)
        context = torch.einsum("b h f m, b h e m -> b h f e", k, v)
        linear_attention = torch.einsum("b h f e, b h f n -> b h e n", context, q)
        out = rearrange(linear_attention, "b h e (i j) -> b (h e) i j", i=i, j=j, h=self.head)
        return self.to_out(out)


def down_sample(dim_in, dim_out):
    return nn.Sequential(Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
                         nn.Conv2d(dim_in * 4, dim_out, kernel_size=(1, 1)))


def up_sample(dim_in, dim_out):
    return nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                         nn.Conv2d(dim_in, dim_out, kernel_size=(3, 3), padding=1))


class UNet(nn.Module):
    def __init__(self, n_bands, ch_per_band, xt_in_ch, out_ch, dim, 
                 dim_multiply=(1, 2, 4, 8), attn_heads=4, attn_head_dim=32,
                 full_attn=(False, False, False, True)):
        super().__init__()
        assert len(dim_multiply) == len(full_attn), "Length of dim_multiply and Length of full_attn must be same"

        self.n_bands = n_bands
        self.ch_per_band = ch_per_band
        self.xt_in_ch = xt_in_ch
        self.out_ch = out_ch
        self.dim = dim
        self.hidden_dims = [self.dim, *map(lambda x: x * self.dim, dim_multiply)]
        self.dim_in_out = list(zip(self.hidden_dims[:-1], self.hidden_dims[1:]))
        self.time_emb_dim = 4 * self.dim
        self.full_attn = full_attn
        self.depth = len(dim_multiply)

        # Time embedding
        positional_encoding = PositionalEncoding(self.dim)
        self.time_mlp = nn.Sequential(
            positional_encoding, nn.Linear(self.dim, self.time_emb_dim),
            nn.GELU(), nn.Linear(self.time_emb_dim, self.time_emb_dim)
        )

        # Layer definition
        self.image_preprocess3d = nn.Sequential(
            nn.Conv3d(self.n_bands, self.dim - self.xt_in_ch, 
                      [self.ch_per_band, 5, 5], padding=[0, 2, 2]),
            nn.BatchNorm3d(self.dim - self.xt_in_ch),
            nn.SiLU(),
        )
        self.init_conv = nn.Sequential(
            # *[ConvBlock(self.dim, self.dim, kernel_size=5) for _ in range(3)],
            *[ConvBlock(self.dim, self.dim, kernel_size=5) for _ in range(2)],
            C3(self.dim, self.dim, n=3),
        )
        self.down_path = nn.ModuleList([])
        self.up_path = nn.ModuleList([])

        # Downward Path layer definition
        resnet_block = partial(ResnetBlock, time_emb_dim=self.time_emb_dim)
        for idx, ((dim_in, dim_out), full_attn_flag) in enumerate(zip(self.dim_in_out, self.full_attn)):
            isLast = idx == (self.depth - 1)
            attention = LinearAttention if not full_attn_flag else Attention
            self.down_path.append(nn.ModuleList([
                resnet_block(dim_in, dim_in),
                resnet_block(dim_in, dim_in),
                attention(dim_in, head=attn_heads, dim_head=attn_head_dim),
                down_sample(dim_in, dim_out) if not isLast else nn.Conv2d(dim_in, dim_out, kernel_size=(3, 3), padding=1)
            ]))

        # Middle layer definition
        mid_dim = self.hidden_dims[-1]
        self.mid_resnet_block1 = resnet_block(mid_dim, mid_dim)
        self.mid_attention = Attention(mid_dim, head=attn_heads, dim_head=attn_head_dim)
        self.mid_resnet_block2 = resnet_block(mid_dim, mid_dim)

        # Upward Path layer definition
        for idx, ((dim_in, dim_out), full_attn_flag) in enumerate(
                zip(reversed(self.dim_in_out), reversed(self.full_attn))):
            isLast = idx == (self.depth - 1)
            attention = LinearAttention if not full_attn_flag else Attention
            self.up_path.append(nn.ModuleList([
                resnet_block(dim_in + dim_out, dim_out),
                resnet_block(dim_in + dim_out, dim_out),
                attention(dim_out, head=attn_heads, dim_head=attn_head_dim),
                up_sample(dim_out, dim_in) if not isLast else nn.Conv2d(dim_out, dim_in, kernel_size=(3, 3), padding=1)
            ]))

        self.final_block = resnet_block(2 * self.dim, self.dim)
        self.final_conv = nn.Sequential(
            # ConvBlock(self.dim, self.dim, kernel_size=3),
            # C3(self.dim, self.dim, n=3),
            # ConvBlock(self.dim, self.dim, kernel_size=3),
            # C3(self.dim, self.dim, n=3),
            ConvBlock(self.dim, self.dim, kernel_size=1),
            C3(self.dim, self.dim, n=3, spatial=False),
            ConvBlock(self.dim, self.dim, kernel_size=1),
            nn.Conv2d(self.dim, self.out_ch, kernel_size=(1, 1))
        )

    def forward(self, x, time, input_image, x_self_cond=None):
        input_image = self.image_preprocess3d(input_image).squeeze(2)
        x = torch.cat([x, input_image], dim=1)
        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)
        concat = list()

        for block1, block2, attn, downsample in self.down_path:
            x = block1(x, t)
            concat.append(x)

            x = block2(x, t)
            x = attn(x) + x
            concat.append(x)

            x = downsample(x)

        x = self.mid_resnet_block1(x, t)
        x = self.mid_attention(x) + x
        x = self.mid_resnet_block2(x, t)

        for block1, block2, attn, upsample in self.up_path:
            x = torch.cat((x, concat.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, concat.pop()), dim=1)
            x = block2(x, t)
            x = attn(x) + x
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_block(x, t)
        return self.final_conv(x)
    

class ImageProcessNet(nn.Module):
    def __init__(self, 
                 n_bands, ch_per_band, 
                 dim, concat_dim_ch: List[int]):
        super().__init__()

        self.n_bands = n_bands
        self.ch_per_band = ch_per_band
        self.dim = dim
        self.out_ch = self.dim * 8
        self.concat_dim_ch = concat_dim_ch
        assert len(self.concat_dim_ch) == 2

        self.preprocess3d = nn.Sequential(
            nn.Conv3d(self.n_bands, self.dim, [self.ch_per_band, 5, 5], padding=[0, 2, 2]),
            nn.BatchNorm3d(self.dim),
            nn.SiLU(),
            Rearrange("b c 1 h w -> b c h w")
        )

        self.direct_down_sample = nn.Sequential(
            *[ConvBlock(self.dim, self.dim, kernel_size=5) for _ in range(4)],
            ConvBlock(self.dim, self.dim, stride=2),
            C3(self.dim, self.dim, n=3),
            ConvBlock(self.dim, self.dim * 2, stride=2),
            C3(self.dim * 2, self.dim * 2, n=3),
        )

        inter_dim = self.dim * 2
        self.conditional_down_sample = nn.ModuleList([])
        for c_dim_ch in self.concat_dim_ch:
            self.conditional_down_sample.append(nn.Sequential(
                    ConvBlock(inter_dim + c_dim_ch, inter_dim * 2, stride=2),
                    C3(inter_dim * 2, inter_dim * 2, n=3)
                )
            )
            inter_dim *= 2

    def forward(self, x, concat_feats: List[torch.Tensor]):
        assert len(concat_feats) == len(self.conditional_down_sample)

        x = self.preprocess3d(x)
        x = self.direct_down_sample(x)
        for c_feats, layer in zip(concat_feats, self.conditional_down_sample):
            x = layer(torch.cat([x, c_feats], dim=1))
        return x
    

class ImageProcessNetV2(nn.Module):
    def __init__(self, 
                 n_bands, 
                 ch_per_band, 
                 dim):
        super().__init__()

        self.n_bands = n_bands
        self.ch_per_band = ch_per_band
        self.dim = dim
        self.out_chs = [self.dim * i for i in [2, 4, 8]]

        self.preprocess3d = nn.Sequential(
            nn.Conv3d(self.n_bands, self.dim, [self.ch_per_band, 5, 5], padding=[0, 2, 2]),
            nn.BatchNorm3d(self.dim),
            nn.SiLU(),
            Rearrange("b c 1 h w -> b c h w")
        )

        self.direct_down_sample = nn.Sequential(
            *[ConvBlock(self.dim, self.dim, kernel_size=5) for _ in range(4)],
            ConvBlock(self.dim, self.dim, stride=2),
            C3(self.dim, self.dim, n=3),
            ConvBlock(self.dim, self.dim * 2, stride=2),
            C3(self.dim * 2, self.dim * 2, n=3),
        )

        inter_dim = self.dim * 2
        self.conditional_down_sample = nn.ModuleList([])
        for _ in range(2):
            self.conditional_down_sample.append(nn.Sequential(
                    ConvBlock(inter_dim, inter_dim * 2, stride=2),
                    C3(inter_dim * 2, inter_dim * 2, n=3)
                )
            )
            inter_dim *= 2

    def forward(self, x):
        x = self.preprocess3d(x)
        x = self.direct_down_sample(x)
        output_list = [x]
        for layer in self.conditional_down_sample:
            x = layer(x)
            output_list.append(x)
        return output_list
    

class YNet(nn.Module):
    def __init__(self, n_bands, ch_per_band, 
                 in_ch, out_ch, dim, 
                 dim_multiply=(1, 2, 4), 
                 attn_heads=4, attn_head_dim=32,
                 full_attn=(False, False, False),
                 use_self_cond=False):
        super().__init__()
        assert len(dim_multiply) == len(full_attn), "Length of dim_multiply and Length of full_attn must be same"

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.dim = dim
        self.hidden_dims = [self.dim, *map(lambda x: x * self.dim, dim_multiply)]
        self.dim_in_out = list(zip(self.hidden_dims[:-1], self.hidden_dims[1:]))
        self.time_emb_dim = 4 * self.dim
        self.full_attn = full_attn
        self.depth = len(dim_multiply)
        self.use_self_cond = use_self_cond

        self.image_process_net = ImageProcessNet(n_bands=n_bands,
                                                 ch_per_band=ch_per_band,
                                                 dim=dim,
                                                 concat_dim_ch=self.hidden_dims[:2])

        # Time embedding
        positional_encoding = PositionalEncoding(self.dim)
        self.time_mlp = nn.Sequential(
            positional_encoding, nn.Linear(self.dim, self.time_emb_dim),
            nn.GELU(), nn.Linear(self.time_emb_dim, self.time_emb_dim)
        )

        # Layer definition
        if self.use_self_cond:
            self.init_conv = nn.Conv2d(self.in_ch + self.out_ch, 
                                       self.dim, kernel_size=(5, 5), padding=2)
        else:
            self.init_conv = nn.Conv2d(self.in_ch, 
                                       self.dim, kernel_size=(5, 5), padding=2)
        
        self.down_path = nn.ModuleList([])
        self.up_path = nn.ModuleList([])

        # Downward Path layer definition
        resnet_block = partial(ResnetBlock, time_emb_dim=self.time_emb_dim)
        for idx, ((dim_in, dim_out), full_attn_flag) in enumerate(zip(self.dim_in_out, self.full_attn)):
            isLast = idx == (self.depth - 1)
            attention = LinearAttention if not full_attn_flag else Attention
            self.down_path.append(nn.ModuleList([
                resnet_block(dim_in, dim_in),
                resnet_block(dim_in, dim_in),
                attention(dim_in, head=attn_heads, dim_head=attn_head_dim),
                down_sample(dim_in, dim_out) if not isLast else nn.Conv2d(dim_in, dim_out, kernel_size=(3, 3), padding=1)
            ]))

        # Middle layer definition
        mid_dim = self.hidden_dims[-1]
        self.mid_resnet_block1 = resnet_block(mid_dim + self.image_process_net.out_ch, mid_dim)
        self.mid_attention = Attention(mid_dim, head=attn_heads, dim_head=attn_head_dim)
        self.mid_resnet_block2 = resnet_block(mid_dim, mid_dim)

        # Upward Path layer definition
        for idx, ((dim_in, dim_out), full_attn_flag) in enumerate(
                zip(reversed(self.dim_in_out), reversed(self.full_attn))):
            isLast = idx == (self.depth - 1)
            attention = LinearAttention if not full_attn_flag else Attention
            self.up_path.append(nn.ModuleList([
                resnet_block(dim_in + dim_out, dim_out),
                resnet_block(dim_in + dim_out, dim_out),
                attention(dim_out, head=attn_heads, dim_head=attn_head_dim),
                up_sample(dim_out, dim_in) if not isLast else nn.Conv2d(dim_out, dim_in, kernel_size=(3, 3), padding=1)
            ]))

        self.final_block = resnet_block(2 * self.dim, self.dim)
        self.final_conv = nn.Sequential(
            ConvBlock(self.dim, self.dim, kernel_size=1),
            C3(self.dim, self.dim, n=3, spatial=False),
            ConvBlock(self.dim, self.dim, kernel_size=1),
            nn.Conv2d(self.dim, self.out_ch, kernel_size=(1, 1))
        )

    def forward(self, x, time, input_image, x_self_cond=None):
        if self.use_self_cond:
            if x_self_cond is None:
                x_self_cond = torch.zeros_like(x)
            x = torch.cat([x, x_self_cond], dim=1)
        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)
        concat = list()
        inter_feats = list()

        for block1, block2, attn, downsample in self.down_path:
            x = block1(x, t)
            concat.append(x)

            x = block2(x, t)
            x = attn(x) + x
            concat.append(x)
            inter_feats.append(x.detach())

            x = downsample(x)

        image_feats = self.image_process_net(input_image, inter_feats[:-1])
        x = torch.cat([x, image_feats], dim=1)

        x = self.mid_resnet_block1(x, t)
        x = self.mid_attention(x) + x
        x = self.mid_resnet_block2(x, t)

        for block1, block2, attn, upsample in self.up_path:
            x = torch.cat((x, concat.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, concat.pop()), dim=1)
            x = block2(x, t)
            x = attn(x) + x
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_block(x, t)
        return self.final_conv(x)
    

class YNetV2(nn.Module):
    def __init__(self, 
                 n_bands, 
                 ch_per_band, 
                 in_ch, 
                 out_ch, 
                 dim, 
                 attn_heads=4, 
                 attn_head_dim=32):
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.dim = dim
        self.dim_multiply = (1, 2, 4)
        self.depth = len(self.dim_multiply)
        self.full_attn = (False, False, False)
        self.hidden_dims = [self.dim, *map(lambda x: x * self.dim, self.dim_multiply)]
        self.dim_in_out = list(zip(self.hidden_dims[:-1], self.hidden_dims[1:]))
        self.time_emb_dim = 4 * self.dim

        self.image_process_net = ImageProcessNetV2(n_bands=n_bands,
                                                    ch_per_band=ch_per_band,
                                                    dim=dim)
        
        self.time_mlp = nn.Sequential(
            PositionalEncoding(self.dim), 
            nn.Linear(self.dim, self.time_emb_dim),
            nn.GELU(), 
            nn.Linear(self.time_emb_dim, self.time_emb_dim)
        )

        self.init_conv = nn.Conv2d(self.in_ch, self.dim, 
                                   kernel_size=(5, 5), padding=2)
        
        self.down_path = nn.ModuleList([])
        self.up_path = nn.ModuleList([])

        resnet_block = partial(ResnetBlock, time_emb_dim=self.time_emb_dim)
        for idx, ((dim_in, dim_out), image_feat_ch, full_attn_flag) in enumerate(zip(self.dim_in_out, 
                                                                                     self.image_process_net.out_chs, 
                                                                                     self.full_attn)):
            isLast = idx == (self.depth - 1)
            attention = LinearAttention if not full_attn_flag else Attention
            self.down_path.append(nn.ModuleList([
                resnet_block(dim_in + image_feat_ch, dim_in),
                resnet_block(dim_in, dim_in),
                attention(dim_in, head=attn_heads, dim_head=attn_head_dim),
                down_sample(dim_in, dim_out) if not isLast else nn.Conv2d(dim_in, dim_out, kernel_size=(3, 3), padding=1)
            ]))

        # Middle layer definition
        mid_dim = self.hidden_dims[-1]
        self.mid_resnet_block1 = resnet_block(mid_dim, mid_dim)
        self.mid_attention = Attention(mid_dim, head=attn_heads, dim_head=attn_head_dim)
        self.mid_resnet_block2 = resnet_block(mid_dim, mid_dim)

        # Upward Path layer definition
        for idx, ((dim_in, dim_out), full_attn_flag) in enumerate(
                zip(reversed(self.dim_in_out), reversed(self.full_attn))):
            isLast = idx == (self.depth - 1)
            attention = LinearAttention if not full_attn_flag else Attention
            self.up_path.append(nn.ModuleList([
                resnet_block(dim_in + dim_out, dim_out),
                resnet_block(dim_in + dim_out, dim_out),
                attention(dim_out, head=attn_heads, dim_head=attn_head_dim),
                up_sample(dim_out, dim_in) if not isLast else nn.Conv2d(dim_out, dim_in, kernel_size=(3, 3), padding=1)
            ]))

        self.final_block = resnet_block(2 * self.dim, self.dim)
        self.final_conv = nn.Sequential(
            ConvBlock(self.dim, self.dim, kernel_size=1),
            C3(self.dim, self.dim, n=3, spatial=False),
            ConvBlock(self.dim, self.dim, kernel_size=1),
            nn.Conv2d(self.dim, self.out_ch, kernel_size=(1, 1))
        )

        self.fast_inference_mode = False
        self.buffer_image = None
        self.buffer_image_feats = None

    def forward(self, x, time, input_image, x_self_cond=None):
        assert x_self_cond is None

        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)

        if self.fast_inference_mode:
            if self.buffer_image is None:
                self.buffer_image = input_image
                self.buffer_image_feats = self.image_process_net(input_image)
            if not torch.allclose(self.buffer_image, input_image):
                self.buffer_image = input_image
                self.buffer_image_feats = self.image_process_net(input_image)
            image_feats = self.buffer_image_feats
        else:
            image_feats = self.image_process_net(input_image)

        concat = []
        for image_f, (block1, block2, attn, downsample) in zip(image_feats, self.down_path):
            x = torch.cat([x, image_f], dim=1)
            x = block1(x, t)
            concat.append(x)

            x = block2(x, t)
            x = attn(x) + x
            concat.append(x)

            x = downsample(x)

        x = self.mid_resnet_block1(x, t)
        x = self.mid_attention(x) + x
        x = self.mid_resnet_block2(x, t)

        for block1, block2, attn, upsample in self.up_path:
            x = torch.cat([x, concat.pop()], dim=1)
            x = block1(x, t)

            x = torch.cat([x, concat.pop()], dim=1)
            x = block2(x, t)
            x = attn(x) + x
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_block(x, t)
        return self.final_conv(x)
    

class FeaturesNet(nn.Module):
    def __init__(self, n_bands, ch_per_band, num_features):
        super().__init__()
        self.preprocess3d = nn.Sequential(
            nn.Conv3d(n_bands, 64, [ch_per_band, 5, 5], padding=[0, 2, 2]),
            nn.BatchNorm3d(64),
            nn.SiLU(),
        )

        # tile_slen is 4 (rather than 2)
        self.downsample_net = nn.Sequential(
            ConvBlock(64, 64, stride=2),  # 2
            C3(64, 64, n=3),
        )

        u_net_layers = [
            ConvBlock(64, 64, kernel_size=5),
            nn.Sequential(*[ConvBlock(64, 64, kernel_size=5) for _ in range(3)]),  # 1
            ConvBlock(64, 128, stride=2),  # 2
            C3(128, 128, n=3),
            ConvBlock(128, 256, stride=2),  # 4
            C3(256, 256, n=3),
            ConvBlock(256, 512, stride=2),
            C3(512, 256, n=3, shortcut=False),
            nn.Upsample(scale_factor=2, mode="nearest"),  # 8
            C3(512, 256, n=3, shortcut=False),
            nn.Upsample(scale_factor=2, mode="nearest"),  # 10
            C3(384, num_features, n=3, shortcut=False),
        ]
        self.u_net = nn.ModuleList(u_net_layers)

    def forward(self, x):
        x = self.preprocess3d(x).squeeze(2)

        save_lst = []
        for i, m in enumerate(self.u_net):
            x = m(x)
            if i in {2, 5}:
                save_lst.append(x)
            if i == 8:
                x = torch.cat((x, save_lst[1]), dim=1)
            if i == 10:
                x = torch.cat((x, save_lst[0]), dim=1)
            if i == 1:
                x = self.downsample_net(x)

        return x


class SimpleNetV1(nn.Module):
    def __init__(self, 
                 n_bands, 
                 ch_per_band, 
                 in_ch, 
                 out_ch, 
                 dim,
                 num_pre_cond_layers,
                 num_post_cond_layers):
        super().__init__()

        self.num_features = 256
        self.features_net = FeaturesNet(n_bands, ch_per_band, self.num_features)

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.dim = dim
        self.time_emb_dim = 4 * self.dim

        positional_encoding = PositionalEncoding(self.dim)
        self.time_mlp = nn.Sequential(
            positional_encoding, 
            nn.Linear(self.dim, self.time_emb_dim),
            nn.GELU(), 
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
        )

        self.init_conv = ConvBlock(self.in_ch, self.dim, kernel_size=1)

        self.pre_cond_layers = nn.ModuleList([ResnetBlock(self.dim, self.dim, 
                                                          time_emb_dim=self.time_emb_dim, 
                                                          spatial=False)
                                              for _ in range(num_pre_cond_layers)])

        self.mid_resnet_block1 = ResnetBlock(self.dim + self.num_features, self.dim, 
                                             time_emb_dim=self.time_emb_dim, spatial=True)
        self.mid_attention = Attention(self.dim, head=4, dim_head=16)
        self.mid_resnet_block2 = ResnetBlock(self.dim, self.dim,
                                             time_emb_dim=self.time_emb_dim, spatial=True)

        self.post_cond_layers = nn.ModuleList([ResnetBlock(self.dim, self.dim, 
                                                           time_emb_dim=self.time_emb_dim, 
                                                           spatial=False)
                                               for _ in range(num_post_cond_layers)])
     
        self.final_block1 = ResnetBlock(2 * self.dim, self.dim, self.time_emb_dim, spatial=True)
        self.final_block2 = ResnetBlock(self.dim, self.dim, self.time_emb_dim, spatial=True)
        self.final_conv = nn.Sequential(
            ConvBlock(self.dim, self.dim, kernel_size=1),
            C3(self.dim, self.dim, n=3, spatial=False),
            ConvBlock(self.dim, self.dim, kernel_size=1),
            nn.Conv2d(self.dim, self.out_ch, kernel_size=(1, 1))
        )

    def forward(self, x, time, input_image, x_self_cond=None):
        assert x_self_cond is None

        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)

        for resnet_block in self.pre_cond_layers:
            x = resnet_block(x, t)

        image_feats = self.features_net(input_image)
        x = torch.cat([x, image_feats], dim=1)
        x = self.mid_resnet_block1(x, t)
        x = self.mid_attention(x) + x
        x = self.mid_resnet_block2(x, t)

        for resnet_block in self.post_cond_layers:
            x = resnet_block(x, t)

        x = torch.cat((x, r), dim=1)
        x = self.final_block1(x, t)
        x = self.final_block2(x, t)
        return self.final_conv(x)
    

class SimpleNetV2(nn.Module):
    def __init__(self, 
                 n_bands, 
                 ch_per_band, 
                 in_ch, 
                 out_ch, 
                 dim,
                 num_cond_layers,
                 spatial_cond_layers):
        super().__init__()

        self.num_features = 256
        self.features_net = FeaturesNet(n_bands, ch_per_band, self.num_features)

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.dim = dim
        self.time_emb_dim = 4 * self.dim

        positional_encoding = PositionalEncoding(self.dim)
        self.time_mlp = nn.Sequential(
            positional_encoding, 
            nn.Linear(self.dim, self.time_emb_dim),
            nn.GELU(), 
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
        )

        self.cond_layers = nn.ModuleList([ResnetBlock(self.in_ch + self.num_features 
                                                      if i == 0 else self.dim, 
                                                      self.dim, 
                                                      time_emb_dim=self.time_emb_dim, 
                                                      spatial=spatial_cond_layers)
                                          for i in range(num_cond_layers)])
     
        self.final_conv = nn.Sequential(
            ConvBlock(self.dim, self.dim, kernel_size=1),
            C3(self.dim, self.dim, n=3, spatial=False),
            ConvBlock(self.dim, self.dim, kernel_size=1),
            nn.Conv2d(self.dim, self.out_ch, kernel_size=(1, 1))
        )

        self.fast_inference_mode = False
        self.buffer_image = None
        self.buffer_image_feats = None

    def forward(self, x, time, input_image, x_self_cond=None):
        assert x_self_cond is None

        if self.fast_inference_mode:
            if self.buffer_image is None:
                self.buffer_image = input_image
                self.buffer_image_feats = self.features_net(input_image)
            if not torch.allclose(self.buffer_image, input_image):
                self.buffer_image = input_image
                self.buffer_image_feats = self.features_net(input_image)
            image_feats = self.buffer_image_feats
        else:
            image_feats = self.features_net(input_image)
        
        x = torch.cat([x, image_feats], dim=1)

        t = self.time_mlp(time)
        for resnet_block in self.cond_layers:
            x = resnet_block(x, t)

        return self.final_conv(x)
