import copy
import torch
from torch import nn
from torch.nn.modules.upsampling import Upsample

from einops import rearrange
from einops.layers.torch import Rearrange

from case_studies.dc2_yolov10.utils.conv_layers import (Conv, C2f, C2fCIB, SPPF, PSA, Concat)
from case_studies.dc2_yolov10.utils.loss import v10DetectLoss
from case_studies.dc2_yolov10.utils.other import box_cxcywh_to_xyxy, make_anchors, dist2bbox


class v10Backbone(nn.Module):
    def __init__(self, n_bands, ch_per_band):
        super().__init__()

        self.n_bands = n_bands
        self.ch_per_band = ch_per_band

        self.preprocess3d = nn.Sequential(
            nn.Conv3d(self.n_bands, 64, [self.ch_per_band, 5, 5], padding=[0, 2, 2]),
            nn.BatchNorm3d(64),
            nn.SiLU(),
            Rearrange("b c 1 h w -> b c h w"),
            # *[Conv(64, 64, k=5) for _ in range(4)],
        )

        self.backbone_blocks = nn.ModuleList([
            Conv(64, 64, k=3, s=2),  # 0: (b, 64, 40, 40)
            C2f(64, 64, n=2, shortcut=True),
            Conv(64, 128, k=3, s=2),  # 2: (b, 128, 20, 20)
            C2f(128, 128, n=4, shortcut=True),
            Conv(128, 256, k=3, s=2),  # 4: (b, 256, 10, 10)
            C2fCIB(256, 256, n=2, shortcut=True),
            SPPF(256, 256, k=5),
            PSA(256, 256),  # 7
        ])

    def forward(self, x):
        x = self.preprocess3d(x)
        output_dict = {}
        for i, m in enumerate(self.backbone_blocks):
            x = m(x)
            if i in [1, 3, 7]:
                output_dict[f"feats_{i}"] = x
        return output_dict


class v10Head(nn.Module):
    def __init__(self):
        super().__init__()

        self.head_blocks = nn.ModuleList([
            Upsample(scale_factor=2, mode="nearest"),  # 0: (b, 256, 20, 20)
            Concat(dimension=1),  # 1: (b, 256 + 128, 20, 20)
            C2f(384, 128, n=2),
            Upsample(scale_factor=2, mode="nearest"),
            Concat(dimension=1),  # 4: (b, 128 + 64, 40, 40)
            C2f(192, 64, n=2),
            Conv(64, 64, k=3, s=2),  # 6: (b, 64, 20, 20)
            Concat(dimension=1),  # 7: (b, 64 + 128, 20, 20)
            C2fCIB(192, 128, n=2, shortcut=True),
            Conv(128, 128, k=3, s=2),  
            Concat(dimension=1),  # 10: (b, 128 + 256, 10, 10)
            C2fCIB(384, 256, n=2, shortcut=True)  # 11
        ])

    def forward(self, backbone_feats):
        x = backbone_feats["feats_7"]
        save_feats = None
        output_list = []
        for i, m in enumerate(self.head_blocks):
            match i:
                case 1:
                    assert isinstance(m, Concat)
                    x = [x, backbone_feats["feats_3"]]
                case 4:
                    assert isinstance(m, Concat)
                    x = [x, backbone_feats["feats_1"]]
                case 7:
                    assert isinstance(m, Concat)
                    x = [x, save_feats]
                case 10:
                    assert isinstance(m, Concat)
                    x = [x, backbone_feats["feats_7"]]
                case _:
                    assert not isinstance(m, Concat)

            x = m(x)

            match i:
                case 2:
                    save_feats = x
                case 5 | 8 | 11:
                    output_list.append(x)
        return output_list
        

class Detect(nn.Module):
    """YOLOv8 Detect head for detection models."""

    def __init__(self, ch):
        """Initializes the YOLOv8 detection layer with specified number of channels."""
        super().__init__()
        self.nc = 1  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)

    def forward_feat(self, x, cv2, cv3):
        y = []
        for i in range(self.nl):
            y.append(torch.cat((cv2[i](x[i]), cv3[i](x[i])), 1))
        return y

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        return self.forward_feat(x, self.cv2, self.cv3)
    

class v10Detect(Detect):
    def __init__(self, ch):
        super().__init__(ch)
        c3 = max(ch[0], min(self.nc, 100))  # channels
        self.cv3 = nn.ModuleList(nn.Sequential(nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)), \
                                               nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)), \
                                                nn.Conv2d(c3, self.nc, 1)) for x in ch)

        self.one2one_cv2 = copy.deepcopy(self.cv2)
        self.one2one_cv3 = copy.deepcopy(self.cv3)
    
    def forward(self, x):
        one2one = self.forward_feat([xi.detach() for xi in x], self.one2one_cv2, self.one2one_cv3)
        one2many = super().forward(x)
        return {"one2many": one2many, "one2one": one2one}


class v10Model(nn.Module):
    def __init__(self, n_bands, ch_per_band):
        super().__init__()

        self.backbone = v10Backbone(n_bands=n_bands,
                                    ch_per_band=ch_per_band)
        self.head = v10Head()
        self.detect = v10Detect(ch=(64, 128, 256))
        self.criterion = v10DetectLoss()

        self.reg_max = 16
        self.register_buffer("stride", torch.tensor([2, 4, 8]))
        self.register_buffer("proj", torch.arange(self.reg_max, dtype=torch.float))
        self.anchors = None
        self.strides = None

    def _infer_images(self, images):
        backbone_feats = self.backbone(images)
        head_output_list = self.head(backbone_feats)
        return self.detect(head_output_list)

    def forward(self, images, gt_cxcywh, gt_mask):
        nn_output = self._infer_images(images)
        gt_xyxy = box_cxcywh_to_xyxy(gt_cxcywh)
        return self.criterion(nn_output, gt_xyxy, gt_mask)

    @torch.inference_mode()
    def sample(self, images):
        nn_output = self._infer_images(images)["one2one"]

        x_cat = torch.cat([rearrange(xi, "b c h w -> b (h w) c") for xi in nn_output], dim=1)
        pred_distri, pred_cls = x_cat.split([16 * 4, 1], dim=-1)

        if self.anchors is None or self.strides is None:
            self.anchors, self.strides = make_anchors(nn_output, self.stride, 0.5)
        
        b, a, c = pred_distri.shape  # batch, anchors, channels
        pred_dist = pred_distri.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_distri.dtype))
        dbox = dist2bbox(pred_dist, self.anchors.unsqueeze(0), xywh=True, dim=-1) * self.strides
        return dbox, pred_cls.sigmoid()
