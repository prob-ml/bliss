# flake8: noqa
from bliss.base_detect.utils.general import LOGGER, colorstr

PREFIX = colorstr("AutoAnchor: ")


def check_anchor_order(m):
    # Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
    a = m.anchors.prod(-1).mean(-1).view(-1)  # mean anchor area per output layer
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da and (da.sign() != ds.sign()):  # same order
        LOGGER.info(f"{PREFIX}Reversing anchor order")  # pylint: disable=W1203
        m.anchors[:] = m.anchors.flip(0)
