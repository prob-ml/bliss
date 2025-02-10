import torch
from torch import nn
import torch.nn.functional as F

from collections import OrderedDict

from bliss.encoder.convnet_layers import C3, ConvBlock, Detect


class FeaturesBackbone(nn.Module):
    def __init__(self, n_bands, ch_per_band):
        super().__init__()
        self.preprocess3d = nn.Sequential(
            nn.Conv3d(n_bands, 64, [ch_per_band, 5, 5], padding=[0, 2, 2]),
            nn.BatchNorm3d(64),
            nn.SiLU(),
        )

        self.generate_f1 = nn.Sequential(
            ConvBlock(64, 64, kernel_size=5),
            nn.Sequential(*[ConvBlock(64, 64, kernel_size=5) for _ in range(3)]),
            ConvBlock(64, 64, stride=2),  # 40x40
            C3(64, 64, n=3),
        )  # f1

        self.f1_to_f2 = nn.Sequential(
            ConvBlock(64, 128, stride=2),  # 20x20
            C3(128, 128, n=3),
        )

        self.f2_to_f3 = nn.Sequential(
            ConvBlock(128, 256, stride=2),  # 10x10
            C3(256, 256, n=3),
        )

        self.f3_to_f4 = nn.Sequential(
            ConvBlock(256, 512, stride=2),  # 5x5
            C3(512, 256, n=3, shortcut=False),
        )

    def forward(self, x):
        x = self.preprocess3d(x).squeeze(2)
        f1 = self.generate_f1(x)
        f2 = self.f1_to_f2(f1)
        f3 = self.f2_to_f3(f2)
        f4 = self.f3_to_f4(f3)

        return {
            "f1": f1,  # (b, 64, 40, 40)
            "f2": f2,  # (b, 128, 20, 20)
            "f3": f3,  # (b, 256, 10, 10)
            "f4": f4,  # (b, 256, 5, 5)
        }
    
    def partial_freeze(self):
        for param in self.preprocess3d.parameters():
            param.requires_grad = False
        for param in self.generate_f1.parameters():
            param.requires_grad = False
        for param in self.f3_to_f4.parameters():
            param.requires_grad = False


class FeatureBackboneOutputHead(nn.Module):
    def __init__(self, out_ch):
        super().__init__()
        self.upsample_layers = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode="nearest"),
            C3(512, 256, n=3, shortcut=False),
            nn.Upsample(scale_factor=2, mode="nearest"),
            C3(384, 256, n=3, shortcut=False),
        ])

        self.output_head = nn.Sequential(
            ConvBlock(256, 256, kernel_size=1),
            C3(256, 256, n=3, spatial=False),
            ConvBlock(256, 256, kernel_size=1),
            Detect(256, out_ch),
        )

    def forward(self, fs):
        x = self.upsample_layers[0](fs["f4"])
        x = torch.cat((x, fs["f3"]), dim=1)
        x = self.upsample_layers[1](x)
        x = self.upsample_layers[2](x)
        x = torch.cat((x, fs["f2"]), dim=1)
        x = self.upsample_layers[3](x)
        return self.output_head(x)


class FPN(nn.Module):
    def __init__(self, feature_backbone: FeaturesBackbone, out_ch: int):
        self.feature_backbone = feature_backbone
        assert isinstance(self.feature_backbone, FeaturesBackbone)
        self.out_ch = out_ch

        self.lateral_conv = nn.ModuleList([
            # the original fpn doesn't use activation in conv
            ConvBlock(256, out_ch, kernel_size=1),
            ConvBlock(128, out_ch, kernel_size=1),
            ConvBlock(64, out_ch, kernel_size=1),
        ])

        self.output_conv = nn.ModuleList([
            ConvBlock(out_ch, out_ch, kernel_size=3),
            ConvBlock(out_ch, out_ch, kernel_size=3),
            ConvBlock(out_ch, out_ch, kernel_size=3),
        ])

        self.fpn_features = ["p1", "p2", "p3"]

    def forward(self, x):
        fs = self.feature_backbone(x)

        out_dict = {}
        start_level = 3
        prev_features = self.lateral_conv[0](fs[f"f{start_level}"])
        out_dict[f"p{start_level}"] = self.output_conv[0](prev_features)
        for i, (lc, oc) in enumerate(
            zip(self.lateral_conv, self.output_conv)
        ):
            if i == 0:
                continue
            top_down_features = F.interpolate(prev_features, scale_factor=2, mode="nearest")
            lateral_features = lc(fs[f"f{start_level - i}"])
            prev_features = lateral_features + top_down_features
            out_dict[f"p{start_level - i}"] = oc(prev_features)

        return OrderedDict([(k, out_dict[k]) for k in self.fpn_features])
