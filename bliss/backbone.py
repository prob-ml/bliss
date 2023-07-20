# flake8: noqa
import contextlib
import logging
import math
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn  # noqa: F401  # pylint: disable=W0611
from yolov5.models.yolo import C3, SPPF, BaseModel, Bottleneck, Concat, Conv, Detect
from yolov5.utils.autoanchor import check_anchor_order, colorstr
from yolov5.utils.general import make_divisible, set_logging

from bliss.modules import Conv3d, Conv3d_down  # pylint: disable=W0611  # noqa: F401

LOGGING_NAME = "yolov5"
set_logging(LOGGING_NAME)  # run before defining LOGGER
LOGGER = logging.getLogger(
    LOGGING_NAME
)  # define globally (used in train.py, val.py, detect.py, etc.)
PREFIX = colorstr("AutoAnchor: ")


class newModel(BaseModel):  # noqa: N801
    def __init__(self, cfg="yolov5s.yaml", ch=3, nc=None, anchors=None):
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub  # pylint: disable=C0415

            self.yaml_file = Path(cfg).name
            with open(cfg, encoding="ascii", errors="ignore") as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        # input channels
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # noqa: WPS429
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(  # pylint: disable=W1203
                f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}"
            )
            self.yaml["nc"] = nc  # override yaml value
        if anchors:
            LOGGER.info(  # pylint: disable=W1203
                f"Overriding model.yaml anchors with anchors={anchors}"
            )
            self.yaml["anchors"] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml["nc"])]  # default names
        self.inplace = self.yaml.get("inplace", True)
        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect)):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)  # pylint: disable=C3001, W0108  # noqa: E731

            # NOTE: 5-D input
            d = 2  # set d = 2 in 5-band case
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, d, s, s))])

            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once

    def forward(self, x, profile=False, visualize=False):
        # pylint: disable=W0237
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4**x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

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


def parse_model(d, ch):  # model_dict, input_channels(3)
    # pylint: disable=W1203, W0123, R0912
    # Parse a YOLOv5 model.yaml dictionary
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw, act = (
        d["anchors"],
        d["nc"],
        d["depth_multiple"],
        d["width_multiple"],
        d.get("activation"),
    )
    if act:
        Conv.default_act = eval(  # noqa: S307, WPS421
            act
        )  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        LOGGER.info(f"{colorstr('activation:')} {act}")  # print
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings  # noqa: S307, WPS421
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings  # noqa: S307, WPS421

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain  # noqa: WPS429, WPS120
        if m in {Conv, Bottleneck, SPPF, C3, torch.nn.ConvTranspose2d}:  # noqa: WPS223
            c1, c2 = ch[f], args[0]  # ch tracks caluculated outputs from all layers
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in {C3}:  # noqa: WPS525
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is torch.nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in {Detect}:  # noqa: WPS525
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)  # noqa: WPS435
        elif m in {Conv3d_down}:  # noqa: WPS525
            c2 = 2
        else:
            c2 = ch[f]

        m_ = (  # noqa: WPS120
            torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        )
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f"{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}")  # print
        save.extend(
            x % i for x in ([f] if isinstance(f, int) else f) if x != -1  # noqa: WPS509
        )  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        if m in {"Conv_Mod"}:  # noqa: WPS525
            ch.append(2)
        else:
            ch.append(c2)

    return torch.nn.Sequential(*layers), sorted(save)
