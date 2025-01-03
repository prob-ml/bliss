# # adapted from https://github.com/taehoon-yoon/Diffusion-Probabilistic-Models/tree/master

# import math

# import torch
# import torch.nn as nn
# from einops import rearrange


# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model):
#         super().__init__()
#         half_d_model = d_model // 2
#         log_denominator = -math.log(10000) / (half_d_model - 1)
#         denominator_ = torch.exp(torch.arange(half_d_model) * log_denominator)
#         self.register_buffer("denominator", denominator_)

#     def forward(self, time):
#         """
#         :param time: shape=(B, )
#         :return: Positional Encoding shape=(B, d_model)
#         """
#         argument = time[:, None] * self.denominator[None, :]  # (B, half_d_model)
#         return torch.cat([argument.sin(), argument.cos()], dim=-1)  # (B, d_model)


# class ResnetBlock(nn.Module):
#     def __init__(self, dim, dim_out, time_emb_dim, dropout=None, groups=32):
#         super().__init__()

#         self.dim, self.dim_out = dim, dim_out

#         self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=dim)
#         self.activation1 = nn.SiLU()
#         self.conv1 = nn.Conv2d(dim, dim_out, kernel_size=(1, 1), padding=0)
#         self.block1 = nn.Sequential(self.norm1, self.activation1, self.conv1)

#         self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))

#         self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=dim_out)
#         self.activation2 = nn.SiLU()
#         self.dropout = nn.Dropout(dropout) if dropout is not None else nn.Identity()
#         self.conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=(1, 1), padding=0)
#         self.block2 = nn.Sequential(self.norm2, self.activation2, self.dropout, self.conv2)

#         self.residual_conv = (
#             nn.Conv2d(dim, dim_out, kernel_size=(1, 1)) if dim != dim_out else nn.Identity()
#         )

#     def forward(self, x, time_emb):
#         hidden = self.block1(x)
#         # add in timestep embedding
#         hidden = hidden + self.mlp(time_emb)[..., None, None]  # (B, dim_out, 1, 1)
#         hidden = self.block2(hidden)
#         return hidden + self.residual_conv(x)


# # class Attention(nn.Module):
# #     def __init__(self, dim, groups=32):
# #         super().__init__()

# #         self.dim, self.dim_out = dim, dim

# #         self.scale = dim ** (-0.5)  # 1 / sqrt(d_k)
# #         self.norm = nn.GroupNorm(num_groups=groups, num_channels=dim)
# #         self.to_qkv = nn.Conv2d(dim, dim * 3, kernel_size=(1, 1))
# #         self.to_out = nn.Conv2d(dim, dim, kernel_size=(1, 1))

# #     def forward(self, x):
# #         b, c, h, w = x.shape
# #         qkv = self.to_qkv(self.norm(x)).chunk(3, dim=1)
# #         # You can think (h*w) as sequence length where c is d_k in <Attention is all you need>
# #         q, k, v = map(lambda t: rearrange(t, "b c h w -> b (h w) c"), qkv)

# #         """
# #         q, k, v shape: (batch, seq_length, d_k)  seq_length = height*width, d_k == c == dim
# #         similarity shape: (batch, seq_length, seq_length)
# #         attention_score shape: (batch, seq_length, seq_length)
# #         attention shape: (batch, seq_length, d_k)
# #         out shape: (batch, d_k, height, width)  d_k == c == dim
# #         return shape: (batch, dim, height, width)
# #         """

# #         similarity = torch.einsum("b i c, b j c -> b i j", q, k)  # Q(K^T)
# #         attention_score = torch.softmax(
# #             similarity * self.scale, dim=-1
# #         )  # softmax(Q(K^T) / sqrt(d_k))
# #         attention = torch.einsum("b i j, b j c -> b i c", attention_score, v)
# #         # attention(Q, K, V) = [softmax(Q(K^T) / sqrt(d_k))]V -> Scaled Dot-Product Attention
# #         out = rearrange(attention, "b (h w) c -> b c h w", h=h, w=w)
# #         return self.to_out(out) + x


# # class ResnetAttentionBlock(nn.Module):
# #     def __init__(self, dim, dim_out, time_emb_dim, dropout=None, groups=32):
# #         super().__init__()

# #         self.dim, self.dim_out = dim, dim_out

# #         self.resnet = ResnetBlock(dim, dim_out, time_emb_dim, dropout, groups)
# #         self.attention = Attention(dim_out, groups)

# #     def forward(self, x, time_emb):
# #         x = self.resnet(x, time_emb)
# #         return self.attention(x)


# class DetectionNet(nn.Module):
#     def __init__(self, xt_in_ch, xt_out_ch, extracted_feats_ch):
#         super().__init__()

#         self.xt_in_ch = xt_in_ch
#         self.xt_out_ch = xt_out_ch
#         self.extracted_feats_ch = extracted_feats_ch
#         self.dim = self.xt_out_ch + self.extracted_feats_ch
#         self.time_emb_dim = 4 * self.dim
#         self.final_out_ch = xt_in_ch

#         self.time_mlp = nn.Sequential(
#             PositionalEncoding(self.dim),
#             nn.Linear(self.dim, self.time_emb_dim),
#             nn.SiLU(),
#             nn.Linear(self.time_emb_dim, self.time_emb_dim),
#         )

#         self.xt_preprocess = nn.Conv2d(self.xt_in_ch, self.xt_out_ch, kernel_size=(1, 1), padding=0)
#         self.xt_process = ResnetBlock(
#             self.xt_out_ch, self.xt_out_ch, self.time_emb_dim, dropout=0, groups=1
#         )
#         self.feats_preprocess = nn.Conv2d(
#             self.extracted_feats_ch, self.extracted_feats_ch, kernel_size=(1, 1), padding=0
#         )
#         self.feats_process = ResnetBlock(
#             self.extracted_feats_ch,
#             self.extracted_feats_ch,
#             self.time_emb_dim,
#             dropout=0,
#             groups=self.extracted_feats_ch // 8,
#         )
#         self.xt_feats_process = nn.ModuleList(
#             [
#                 ResnetBlock(self.dim, self.dim, self.time_emb_dim, dropout=0, groups=self.dim // 8)
#                 for _ in range(4)
#             ]
#         )
#         self.final_process = nn.Sequential(
#             nn.GroupNorm(self.dim // 8, self.dim),
#             nn.SiLU(),
#             nn.Conv2d(self.dim, self.dim // 2, kernel_size=(1, 1), padding=0),
#             nn.GroupNorm(self.dim // (8 * 2), self.dim // 2),
#             nn.SiLU(),
#             nn.Conv2d(self.dim // 2, self.final_out_ch, kernel_size=(1, 1), padding=0),
#         )

#     def forward(self, x, time, extracted_feats, self_cond=None):
#         assert self_cond is None
#         t = self.time_mlp(time)
#         x = self.xt_preprocess(x)
#         x = self.xt_process(x, t)
#         extracted_feats = self.feats_preprocess(extracted_feats)
#         extracted_feats = self.feats_process(extracted_feats, t)
#         x = torch.cat([x, extracted_feats], dim=1)
#         for layer in self.xt_feats_process:
#             x = layer(x, t)
#         return self.final_process(x)
