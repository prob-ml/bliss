"""Backbone for BLISS Encoder which adapts the yolov5 DetectionModel class. We
extend this model to allow for 3-D convolution layers and 5-D input support.
See: https://github.com/ultralytics/yolov5

We extend this class to allow for the separation of all band-specific features
(e.g. image, background, deconvolved image, and PSF parameters) in the input
to our model. We format our input to the BLISS encoder as:

BatchSize x Bands x Features x H x W

From here, we extend yolov5 to permit the addition of a 3-D convolution module
that can learn across features. Changes are specifically made to the
DetectionModel constructor, associated methods and parse_model.
"""

# flake8: noqa # pylint: disable-all
import contextlib
import math
from copy import deepcopy
from pathlib import Path

import torch
import yaml
from torch import nn  # needed to parse config
from yolov5.models.yolo import C3, SPPF, BaseModel, Bottleneck, Concat, Conv, Detect
from yolov5.utils.autoanchor import check_anchor_order
from yolov5.utils.general import make_divisible

# needed to parse config
from bliss.modules import Conv3d, Conv3d_down


class Backbone(BaseModel):
    def __init__(self, cfg="yolov5s.yaml", ch=3, nc=None, anchors=None, n_imgs=2):
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding="ascii", errors="ignore") as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        # input channels
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)
        self.num_imgs = n_imgs
        self.model, self.save = parse_model(
            deepcopy(self.yaml), ch=[ch], n_imgs=self.num_imgs
        )  # model, savelist
        self.names = [str(i) for i in range(self.yaml["nc"])]  # default names
        self.inplace = self.yaml.get("inplace", True)
        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect)):
            s = 256  # 2x min stride
            m.inplace = self.inplace

            # NOTE: 5-D input
            d = self.num_imgs  # set d = 2 in 5-band case
            m.stride = torch.tensor(
                [s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, d, s, s))]
            )

            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once

    def forward(self, x, profile=False, visualize=False):
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5 : 5 + m.nc] += (
                math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())
            )  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


def parse_model(d, ch, n_imgs):  # model_dict, input_channels(3)
    # Parse a YOLOv5 model.yaml dictionary
    anchors, nc, gd, gw, act = (
        d["anchors"],
        d["nc"],
        d["depth_multiple"],
        d["width_multiple"],
        d.get("activation"),
    )
    if act:
        Conv.default_act = eval(
            act
        )  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in {Conv, Bottleneck, SPPF, C3, torch.nn.ConvTranspose2d}:
            c1, c2 = ch[f], args[0]  # ch tracks caluculated outputs from all layers
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m is C3:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is torch.nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m in {Conv3d_down}:
            c2 = n_imgs
        else:
            c2 = ch[f]

        if i == 0:
            args[0] = ch[0]

        m_add = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        np = sum(x.numel() for x in m_add.parameters())  # number params
        m_add.i, m_add.f, m_add.type, m_add.np = (
            i,
            f,
            t,
            np,
        )  # attach index, 'from' index, type, number params
        save.extend(
            x % i for x in ([f] if isinstance(f, int) else f) if x != -1
        )  # append to savelist
        layers.append(m_add)
        if i == 0:
            continue
        ch.append(c2)

    return torch.nn.Sequential(*layers), sorted(save)
