import math
import torch
import torch.nn as nn

from einops import rearrange

from bliss.encoder.convnet_layers import C3, ConvBlock


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0

    def forward(self, t: torch.Tensor):  # (b,)
        assert t.ndim == 1
        assert t.dtype == torch.float
        # assert t.max() <= 1.0 and t.min() >= 0.0  # dopri5 will generate t > 1.0 due to adaptive steps
        t = t * 1000
        # standard DDPM sinusoidal embedding
        half = self.dim // 2
        device = t.device
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(0, half, device=device).float() / half
        )
        args = t.float()[:, None] * freqs[None]  # (b, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (b, dim)
        return emb


def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)


def upsample(in_ch, out_ch):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        conv3x3(in_ch, out_ch)
    )


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, cond_ch=None, dropout=0.0):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.act = nn.SiLU()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # time embedding -> scale, shift (FiLM)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch * 2)
        )

        # optional spatial conditioning (project cond map to out_ch, add as bias)
        self.has_cond = cond_ch is not None
        self.cond_proj = nn.Conv2d(cond_ch, out_ch, kernel_size=1) if self.has_cond else None

        # skip when in_ch != out_ch
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb, cond_map=None):
        h = self.conv1(self.act(self.norm1(x)))
        # FiLM from time embedding
        t = self.time_mlp(t_emb)  # (b, 2*out_ch)
        scale, shift = t.chunk(2, dim=1)        # (b, out_ch), (b, out_ch)
        scale = scale[..., None, None]          # (b, out_ch, 1, 1)
        shift = shift[..., None, None]          # (b, out_ch, 1, 1)
        if self.has_cond and (cond_map is not None):
            h = h + self.cond_proj(cond_map)
        # second conv with FiLM
        h = self.conv2(self.act(self.norm2(h)) * (1 + scale) + shift)
        h = self.dropout(h)
        return h + self.skip(x)


class ImgBackbone(nn.Module):
    """
    Simple CNN pyramid:
    80x80 -> 40x40 -> 20x20 -> 10x10 -> 5x5
    We return features at {20, 10, 5} for injection at matching U-Net stages.
    """
    def __init__(self, n_bands, ch_per_band, base_ch):
        super().__init__()
        
        self.preprocess3d = nn.Sequential(
            nn.Conv3d(n_bands, base_ch, [ch_per_band, 5, 5], padding=[0, 2, 2]),
            nn.BatchNorm3d(base_ch),
            nn.SiLU(),
        )
        self.s2_40 = nn.Sequential(  # 80 -> 40
            ConvBlock(base_ch, base_ch, kernel_size=5),
            nn.Sequential(*[ConvBlock(base_ch, base_ch, kernel_size=5) 
                            for _ in range(3)]),
            ConvBlock(base_ch, base_ch, stride=2),
        )
        self.s2_20 = nn.Sequential(  # 40 -> 20
            C3(base_ch, base_ch, n=3),
            ConvBlock(base_ch, base_ch * 2, stride=2),
        )
        self.s2_10 = nn.Sequential(  # 20 -> 10
            C3(base_ch * 2, base_ch * 2, n=3),
            ConvBlock(base_ch * 2, base_ch * 4, stride=2),
        )
        self.s2_5 = nn.Sequential(   # 10 -> 5
            C3(base_ch * 4, base_ch * 4, n=3),
            ConvBlock(base_ch * 4, base_ch * 5, stride=2),
        )

        # channels at each scale:
        self.c20c10c05 = [base_ch * 2, base_ch * 4, base_ch * 5]

    def forward(self, img):
        prep_img = self.preprocess3d(img).squeeze(2)  # (b, base_ch, 80, 80)
        f40 = self.s2_40(prep_img)
        f20 = self.s2_20(f40)
        f10 = self.s2_10(f20)
        f05 = self.s2_5(f10)
        return {"20": f20, "10": f10, "5": f05}


class UShapeHead(nn.Module):
    def __init__(self, 
                 x_in_ch, 
                 x_out_ch, 
                 backbone_c20c10c05,
                 base_ch=64, 
                 time_dim=256, 
                 dropout=0.0):
        super().__init__()

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )

        c20 = base_ch * 1   # 64
        c10 = base_ch * 2   # 128
        c05 = base_ch * 4   # 256

        self.in_conv = conv3x3(x_in_ch, c20)

        backbone_c20, backbone_c10, backbone_c05 = backbone_c20c10c05

        # 20x20 blocks
        self.d20_1 = ResBlock(c20, c20, time_dim, cond_ch=backbone_c20, dropout=dropout)
        self.d20_2 = ResBlock(c20, c20, time_dim, cond_ch=backbone_c20, dropout=dropout)
        self.down_20_to_10 = conv3x3(c20, c10, stride=2)  # 20 -> 10

        # 10x10 blocks
        self.d10_1 = ResBlock(c10, c10, time_dim, cond_ch=backbone_c10, dropout=dropout)
        self.d10_2 = ResBlock(c10, c10, time_dim, cond_ch=backbone_c10, dropout=dropout)
        self.down_10_to_05 = conv3x3(c10, c05, stride=2)  # 10 -> 5

        # bottleneck @ 5x5
        self.mid_1 = ResBlock(c05, c05, time_dim, cond_ch=backbone_c05, dropout=dropout)
        self.mid_2 = ResBlock(c05, c05, time_dim, cond_ch=backbone_c05, dropout=dropout)

        self.up_05_to_10 = upsample(c05, c10)
        self.u10_1 = ResBlock(c10 + c10, c10, time_dim, cond_ch=backbone_c10, dropout=dropout)
        self.u10_2 = ResBlock(c10, c10, time_dim, cond_ch=backbone_c10, dropout=dropout)

        self.up_10_to_20 = upsample(c10, c20)
        self.u20_1 = ResBlock(c20 + c20, c20, time_dim, cond_ch=backbone_c20, dropout=dropout)
        self.u20_2 = ResBlock(c20, c20, time_dim, cond_ch=backbone_c20, dropout=dropout)

        self.out_norm = nn.GroupNorm(32, c20)
        self.out_act  = nn.SiLU()
        self.out_conv = nn.Sequential(
            conv3x3(c20, c20),
            nn.SiLU(),
            nn.Conv2d(c20, c20, kernel_size=1),
            nn.SiLU(),
        )
        self.final_layer = nn.Conv2d(c20, x_out_ch, kernel_size=1)
        nn.init.constant_(self.final_layer.weight, 0.0)
        nn.init.constant_(self.final_layer.bias, 0.0)

    def forward(self, xt, t, img_cond):
        """
        xt: (b, x_in_ch, 20, 20)
        t: (b, ) float in [0, 1]
        img_cond: dict of tensors
        returns (b, x_out_ch, 20, 20)
        """
        t_emb = self.time_embed(t)

        # down
        h0 = self.in_conv(xt)
        h1 = self.d20_1(h0, t_emb, img_cond["20"])
        h2 = self.d20_2(h1, t_emb, img_cond["20"])
        d1 = self.down_20_to_10(h2)

        h3 = self.d10_1(d1, t_emb, img_cond["10"])
        h4 = self.d10_2(h3, t_emb, img_cond["10"])
        d2 = self.down_10_to_05(h4)

        m1 = self.mid_1(d2, t_emb, img_cond["5"])
        m2 = self.mid_2(m1, t_emb, img_cond["5"])

        # up
        u1 = self.up_05_to_10(m2)
        u1 = torch.cat([u1, h4], dim=1)
        u1 = self.u10_1(u1, t_emb, img_cond["10"])
        u1 = self.u10_2(u1, t_emb, img_cond["10"])

        u2 = self.up_10_to_20(u1)
        u2 = torch.cat([u2, h2], dim=1)
        u2 = self.u20_1(u2, t_emb, img_cond["20"])
        u2 = self.u20_2(u2, t_emb, img_cond["20"])

        out = self.out_conv(self.out_act(self.out_norm(u2)))
        return self.final_layer(out)


class UUNet(nn.Module):
    def __init__(self, 
                 img_n_bands, 
                 img_ch_per_band, 
                 img_backbone_base_ch,
                 ns_ch, 
                 ns_params, 
                 ns_unet_base_ch,
                 other_ch,
                 other_params,
                 other_unet_base_ch,
                ):
        super().__init__()

        self.img_backbone = ImgBackbone(n_bands=img_n_bands,
                                        ch_per_band=img_ch_per_band,
                                        base_ch=img_backbone_base_ch)
        self.ns_unet = UShapeHead(x_in_ch=ns_ch,
                                  x_out_ch=ns_params,
                                  backbone_c20c10c05=self.img_backbone.c20c10c05,
                                  base_ch=ns_unet_base_ch,
                                  time_dim=ns_unet_base_ch * 4)
        self.other_unet = UShapeHead(x_in_ch=other_ch + ns_ch,
                                     x_out_ch=other_params,
                                     backbone_c20c10c05=self.img_backbone.c20c10c05,
                                     base_ch=other_unet_base_ch,
                                     time_dim=other_unet_base_ch * 4)
        self.ns_ch = ns_ch
        self.other_ch = other_ch
        
        self.fast_inference_mode = False
        self.buffer_img = None
        self.buffer_img_cond = None

    def enter_fast_inference(self):
        self.fast_inference_mode = True
        assert self.buffer_img is None
        assert self.buffer_img_cond is None

    def exit_fast_inference(self):
        self.fast_inference_mode = False
        assert self.buffer_img is not None
        self.buffer_img = None
        assert self.buffer_img_cond is not None
        self.buffer_img_cond = None

    @classmethod
    def _interleave_ns_and_other(cls, ns_x: torch.Tensor, other_x: torch.Tensor):
        ns1_x, ns2_x = torch.chunk(ns_x, 2, dim=1)
        other1_x, other2_x = torch.chunk(other_x, 2, dim=1)
        return torch.cat([ns1_x, other1_x, ns2_x, other2_x], dim=1)
        
    def forward(self, *, n_sources_xt, other_xt, t, image, n_sources_x1):
        assert n_sources_xt.ndim == 4
        assert n_sources_xt.shape[-1] == self.ns_ch
        assert n_sources_xt.shape == n_sources_x1.shape
        assert other_xt.ndim == 4
        assert other_xt.shape[-1] == self.other_ch

        n_sources_xt = rearrange(n_sources_xt, "b h w k -> b k h w")
        other_xt = rearrange(other_xt, "b h w k -> b k h w")
        n_sources_x1 = rearrange(n_sources_x1, "b h w k -> b k h w")

        img_cond = self.img_backbone(image)
        pred_ns_cat = self.ns_unet(xt=n_sources_xt, t=t, img_cond=img_cond)
        pred_other_cat = self.other_unet(xt=self._interleave_ns_and_other(n_sources_x1, other_xt),
                                         t=t,
                                         img_cond=img_cond)
        pred_ns_cat1, pred_ns_cat2 = torch.chunk(pred_ns_cat, 2, dim=1)
        pred_other_cat1, pred_other_cat2 = torch.chunk(pred_other_cat, 2, dim=1)
        pred_cat1 = torch.cat([pred_ns_cat1, pred_other_cat1], dim=1)
        pred_cat2 = torch.cat([pred_ns_cat2, pred_other_cat2], dim=1)
        return rearrange(pred_cat1, "b k h w -> b h w k"), rearrange(pred_cat2, "b k h w -> b h w k")
    
    @torch.inference_mode()
    def get_image_cond(self, image):
        if self.fast_inference_mode:
            if self.buffer_img is None:
                self.buffer_img = image
                self.buffer_img_cond = self.img_backbone(image)
            assert torch.allclose(self.buffer_img, image), "in fast inference mode, we get different images"
            img_cond = self.buffer_img_cond
        else:
            img_cond = self.img_backbone(image)
        return img_cond
    
    @torch.inference_mode()
    def calculate_n_sources_cat(self, *, n_sources_xt, t, image):
        assert n_sources_xt.ndim == 4
        assert n_sources_xt.shape[-1] == self.ns_ch

        n_sources_xt = rearrange(n_sources_xt, "b h w k -> b k h w")

        img_cond = self.get_image_cond(image)
        pred_ns_cat = self.ns_unet(xt=n_sources_xt, t=t, img_cond=img_cond)
        pred_ns_cat1, pred_ns_cat2 = torch.chunk(pred_ns_cat, 2, dim=1)
        return rearrange(pred_ns_cat1, "b k h w -> b h w k"), rearrange(pred_ns_cat2, "b k h w -> b h w k")
    
    @torch.inference_mode()
    def calculate_other_cat(self, *, other_xt, t, image, n_sources_x1):
        assert other_xt.ndim == 4
        assert other_xt.shape[-1] == self.other_ch

        other_xt = rearrange(other_xt, "b h w k -> b k h w")
        n_sources_x1 = rearrange(n_sources_x1, "b h w k -> b k h w")

        img_cond = self.get_image_cond(image)
        pred_other_cat = self.other_unet(xt=self._interleave_ns_and_other(n_sources_x1, other_xt),
                                         t=t,
                                         img_cond=img_cond)
        pred_other_cat1, pred_other_cat2 = torch.chunk(pred_other_cat, 2, dim=1)
        return rearrange(pred_other_cat1, "b k h w -> b h w k"), rearrange(pred_other_cat2, "b k h w -> b h w k")
