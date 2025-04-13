import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Mlp
from timm.layers import trunc_normal_

from case_studies.dc2_mdt.utils.convnet_layers import FeaturesNet


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Attention(nn.Module):
    def __init__(self, 
                 *,
                 dim, 
                 num_heads, 
                 num_patches,
                 qkv_bias=False, 
                 attn_drop=0.0, 
                 proj_drop=0.0):
        super().__init__()

        self.num_heads = num_heads
        assert dim % num_heads == 0
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rel_pos_bias = RelativePositionBias(
            window_size=[int(num_patches ** 0.5), int(num_patches ** 0.5)], 
            num_heads=num_heads
        )

    def get_masked_rel_bias(self, B, ids_keep):
        # get masked rel_pos_bias
        rel_pos_bias = self.rel_pos_bias()
        rel_pos_bias = rel_pos_bias.unsqueeze(dim=0).repeat(B, 1, 1, 1)

        rel_pos_bias_masked = torch.gather(
            rel_pos_bias, 
            dim=2, 
            index=ids_keep.unsqueeze(dim=1).unsqueeze(dim=-1).repeat(1, rel_pos_bias.shape[1], 1, rel_pos_bias.shape[-1]))
        rel_pos_bias_masked = torch.gather(
            rel_pos_bias_masked, 
            dim=3, 
            index=ids_keep.unsqueeze(dim=1).unsqueeze(dim=2).repeat(1, rel_pos_bias.shape[1], ids_keep.shape[1], 1))
        return rel_pos_bias_masked

    def forward(self, x, ids_keep=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, C // num_heads)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        if ids_keep is not None:
            rp_bias = self.get_masked_rel_bias(B, ids_keep)
        else:
            rp_bias = self.rel_pos_bias()
        attn += rp_bias
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class RelativePositionBias(nn.Module):
    # https://github.com/microsoft/unilm/blob/master/beit/modeling_finetune.py
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (
            2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads))

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"), dim=0)  # (2, h, w)
        coords_flatten = torch.flatten(coords, 1)  # (2, h*w)
        relative_coords = coords_flatten[:, :, None] - \
                          coords_flatten[:, None, :]  # (2, h*w, h*w)
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()   # (h*w, h*w, 2)
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = \
            torch.zeros(size=(window_size[0] * window_size[1],) * 2, 
                        dtype=relative_coords.dtype)
        relative_position_index = relative_coords.sum(-1)  # (h*w, h*w)

        self.register_buffer("relative_position_index", relative_position_index)

        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self):
        relative_position_bias = \
            self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1], -1)  # (h*w, h*w, num_heads)
        # (num_heads, h*w, h*w)
        return relative_position_bias.permute(2, 0, 1).contiguous()


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0,
                                                 end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


#################################################################################
#                                 Core MDT Model                                #
#################################################################################

class MDTBlock(nn.Module):
    """
    A MDT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio, skip=False, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            dim=hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        def approx_gelu(): return nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size,
                       hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.skip_linear = nn.Linear(2 * hidden_size, hidden_size) if skip else None

    def forward(self, x, c, skip=None, ids_keep=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * \
            self.attn(modulate(self.norm1(x), shift_msa, scale_msa), 
                      ids_keep=ids_keep)
        x = x + gate_mlp.unsqueeze(1) * \
            self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of MDT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class MDTv2(nn.Module):
    """
    Masked Diffusion Transformer v2.
    """

    def __init__(
        self,
        *,
        image_n_bands,
        image_ch_per_band,
        image_feats_ch,
        input_size,
        patch_size,
        in_channels,
        hidden_size,
        depth,
        num_heads,
        decode_layers,
        mlp_ratio,
        learn_sigma,
        mask_ratio,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.decode_layers = int(decode_layers)

        self.image_features_net = FeaturesNet(image_n_bands, image_ch_per_band, image_feats_ch)

        assert hidden_size % 8 == 0
        self.x_embedder = PatchEmbed(input_size, 
                                     patch_size, 
                                     in_channels, 
                                     (hidden_size // 8) * 1, 
                                     bias=True)
        self.image_embedder = PatchEmbed(input_size,
                                         patch_size,
                                         image_feats_ch,
                                         (hidden_size // 8) * 7,
                                         bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        num_patches = self.x_embedder.num_patches
        # Will use learnbale sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(
            1, num_patches, hidden_size), requires_grad=True)
        
        assert (depth - self.decode_layers) % 2 == 0
        half_depth = (depth - self.decode_layers) // 2
        self.half_depth = half_depth
        
        self.en_inblocks = nn.ModuleList([
            MDTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, num_patches=num_patches) 
            for _ in range(half_depth)
        ])
        self.en_outblocks = nn.ModuleList([
            MDTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, num_patches=num_patches, skip=True) 
            for _ in range(half_depth)
        ])
        self.de_blocks = nn.ModuleList([
            MDTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, num_patches=num_patches, skip=True) 
            for _ in range(self.decode_layers)
        ])
        self.sideblocks = nn.ModuleList([
            MDTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, num_patches=num_patches) 
            for _ in range(1)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.decoder_pos_embed = nn.Parameter(torch.zeros(
            1, num_patches, hidden_size), requires_grad=True)
        if mask_ratio is not None:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            self.mask_ratio = float(mask_ratio)
        else:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size), 
                                           requires_grad=False)
            self.mask_ratio = None
        
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.en_inblocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        for block in self.en_outblocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        for block in self.de_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        for block in self.sideblocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        if self.mask_ratio is not None:
            torch.nn.init.normal_(self.mask_token, std=.02)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def forward_side_interpolater(self, x, c, mask, ids_restore):
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        
        # add pos embed
        x = x + self.decoder_pos_embed

        # pass to the basic block
        x_before = x
        for sideblock in self.sideblocks:
            x = sideblock(x, c, ids_keep=None)
        
        # masked shortcut
        mask = mask.unsqueeze(dim=-1)
        x = x * mask + (1 - mask) * x_before

        return x

    def forward(self, x, t, image, enable_mask=False):
        """
        Forward pass of MDT.
        x: (N, C, H, W) tensor of spatial inputs (catalog)
        t: (N,) tensor of diffusion timesteps
        image: astronomical image
        enable_mask: Use mask latent modeling
        """
        image_feats = self.image_features_net(image)
        image_feats = self.image_embedder(image_feats)
        x = self.x_embedder(x)
        x = torch.cat([x, image_feats], dim=-1)
        x = x + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2

        t = self.t_embedder(t)  # (N, D)

        input_skip = x

        masked_stage = False
        skips = []
        # masking op for training
        if self.mask_ratio is not None and enable_mask:
            # masking: length -> length * mask_ratio
            rand_mask_ratio = torch.rand(1, device=x.device)  # noise in [0, 1]
            rand_mask_ratio = rand_mask_ratio * 0.2 + self.mask_ratio # mask_ratio, mask_ratio + 0.2 
            x, mask, ids_restore, ids_keep = self.random_masking(x, rand_mask_ratio)
            masked_stage = True

        for block in self.en_inblocks:
            if masked_stage:
                x = block(x, t, ids_keep=ids_keep)
            else:
                x = block(x, t, ids_keep=None)
            skips.append(x)

        for block in self.en_outblocks:
            if masked_stage:
                x = block(x, t, skip=skips.pop(), ids_keep=ids_keep)
            else:
                x = block(x, t, skip=skips.pop(), ids_keep=None)

        if self.mask_ratio is not None and enable_mask:
            x = self.forward_side_interpolater(x, t, mask, ids_restore)
            masked_stage = False
        else:
            # add pos embed
            x = x + self.decoder_pos_embed

        for i in range(len(self.de_blocks)):
            block = self.de_blocks[i]
            this_skip = input_skip
            x = block(x, t, skip=this_skip, ids_keep=None)

        x = self.final_layer(x, t)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   MDTv2 Configs                               #
#################################################################################

def MDTv2_XL_2(**kwargs):
    return MDTv2(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def MDTv2_L_2(**kwargs):
    return MDTv2(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def MDTv2_B_2(**kwargs):
    return MDTv2(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def MDTv2_S_2(**kwargs):
    return MDTv2(depth=10, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

