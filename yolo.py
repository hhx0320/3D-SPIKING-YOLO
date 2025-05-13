# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""
YOLO-specific modules.

Usage:
    $ python models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import contextlib
import math
import os
import platform
import sys
import copy
from pathlib import Path

import torch
import torch.nn as nn
timestep=4
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != "Windows":
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import (
    C3,
    C3SPP,
    C3TR,
    SPP,
    SPPF,
    Bottleneck,
    BottleneckCSP,
    C3Ghost,
    C3x,
    Classify,
    Concat,
    Contract,
    Conv,
    CrossConv,
    DetectMultiBackend,
    DWConv,
    DWConvTranspose2d,
    Expand,
    Focus,
    GhostBottleneck,
    GhostConv,
    Proto, mem_update,
)
from models.experimental import MixConv2d
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, colorstr, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (
    fuse_conv_and_bn,
    initialize_weights,
    model_info,
    profile,
    scale_img,
    select_device,
    time_sync,
)

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


class Detect(nn.Module):
    # YOLOv5 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        """Initializes YOLOv5 detection layer with specified classes, anchors, channels, and inplace operations."""
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)
        self.scale = nn.Parameter(torch.ones([3, timestep]))
    def forward(self, x):
        """Processes input through YOLOv5 layers, altering shape for detection: `x(bs, 3, ny,  nx, 85)`."""
        z = []
        # if not self.training:
        for i in range(self.nl):
            bs, _, ny, nx = x[i].shape
            # x(bs,255,20,20) to x(bs,3,20,20,85)

            x[i] = x[i].reshape([timestep, -1, _, ny, nx ])

            out = 0
            for j in range(timestep):
                out += x[i][j] * self.scale[i][j]

            x[i] = out
            # x[i] = x[i].mean(0)
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape
            # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                if isinstance(self, Segment):  # (boxes + masks)
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                else:  # Detect (boxes only)
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, "1.10.0")):
        """Generates a mesh grid for anchor boxes with optional compatibility for torch versions < 1.10."""
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing="ij") if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class Segment(Detect):
    # YOLOv5 Segment head for segmentation models
    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
        """Initializes YOLOv5 Segment head with options for mask count, protos, and channel adjustments."""
        super().__init__(nc, anchors, ch, inplace)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

    def forward(self, x):
        """Processes input through the network, returning detections and prototypes; adjusts output based on
        training/export mode.
        """
        p = self.proto(x[0])
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1])

import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """YOLOv5 base model."""

    def forward(self, x, profile=False, visualize=False):
        """Executes a single-scale inference or training pass on the YOLOv5 base model, with options for profiling and
        visualization.
        """
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_once(self, imgs, profile=False, visualize=False,con1=None,con2=None,bn1=None,bn2=None):
        """Performs a forward pass on the YOLOv5 model, enabling profiling and feature visualization options."""
        y, dt = [], []  # outputs
        # x=imgs
        # x = bn2(con2(imgs))

        x=imgs.unsqueeze(0).repeat(timestep,1,1,1,1).flatten(0, 1)
        # x = self.conv1_s(x)

        # imgscopy = bn1[timestep - 1](con1[timestep - 1](imgs))
        #
        # # #
        # sigmoid = nn.Sigmoid()
        #
        #
        # lano = nn.LayerNorm(normalized_shape=[3, imgs.size(2), imgs.size(3)]).to(imgs.device)
        # # imgsfu=imgs.clone()
        #
        # # # n31 = [0.8662419354838822, 0.8233838709677542, 0.7900258064516253, 0.760667741935496, 0.7340096774193664,
        # # #        0.7097516129032364, 0.6863935483871064, 0.6642354838709763, 0.6427774193548461, 0.6217193548387158,
        # # #        0.6015612903225854, 0.5815032258064551, 0.5617451612903247, 0.5421870967741942, 0.5212290322580639,
        # # #        0.5004709677419337, 0.47951290322580337, 0.4585548387096731, 0.4369967741935429, 0.4153387096774127,
        # # #        0.39318064516128254, 0.37072258064515246, 0.34786451612902236, 0.3241064516128924, 0.2995483870967625,
        # # #        0.27339032258063284, 0.24513225806450334, 0.2140741935483781, 0.17791612903225396, 0.1303580645161311,
        # # #        0.0]
        # # # n19 = [0.8370684210526437, 0.7810368421052757, 0.7353052631579065, 0.6955736842105367, 0.6590421052631664,
        # # #         0.6244105263157961, 0.5910789473684255, 0.558147368421055, 0.5254157894736843, 0.49188421052631376,
        # # #         0.45815263157894326, 0.42432105263157277, 0.38908947368420244, 0.35325789473683217, 0.3148263157894622,
        # # #         0.2736947368420925, 0.22716315789472616, 0.17013157894736552, 0.0]
        # # n14 = [0.8145714285714412, 0.7472428571428692, 0.6929142857142959, 0.6429857142857222, 0.596157142857148,
        # #               0.5513285714285736, 0.5053999999999994, 0.4598714285714251, 0.41354285714285094, 0.3656142857142769,
        # #               0.3142857142857033, 0.2575571428571302, 0.18842857142856598, 0.0]
        # # # n9 = [0.7743888888889014, 0.6856777777777879, 0.6101666666666729, 0.5398555555555573, 0.46854444444444177,
        # # #        0.39583333333332643, 0.31822222222221164, 0.22701111111110026, 0.0]
        # # #
        # # # one = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
        # # # two = [1, 2, 5, 6, 9, 10, 13, 14, 17, 18, 21, 22, 25, 26, 29, 30]
        # # # three = [3, 4, 5, 6, 11, 12, 13, 14, 19, 20, 21, 22, 27, 28, 29, 30]
        # # # four = [7, 8, 9, 10, 11, 12, 13, 14, 23, 24, 25, 26, 27, 28, 29, 30]
        # # # five = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
        # # lis = []
        # lis = torch.zeros(timestep-1 , imgs.size(0), 3, imgs.size(2), imgs.size(3))
        # device = imgs.device
        # lis = lis.to(device)
        # imgs = lano(imgs)
        # imgs = sigmoid(imgs)
        #
        # # n=0.5
        # for i in range(timestep-1):
        #
        #     z = (i + 1.0)
        #     z = 1. - z / (timestep-1)
        #     z = z.__float__()
        #     x = torch.where( imgs> z, 1., 0)
        #     imgs -=imgs * x
        #     lis[i] += bn1[0](con1[0](x))
        #     # n=n/2
        # #     # if i in one:
        # #     #     lis[0] += x
        # #     #
        # #     # if i in two:
        # #     #     lis[1] += x
        # #     #
        # #     # if i in three:
        # #     #     lis[2] += x
        # #     #
        # #     # if i in four:
        # #     #     lis[3] += x
        # #     #
        # #     # if i in five:
        # #     #     lis[4] += x
        # #     # imgs -= n * x
        # #     # n=n/2
        # #     lis.append(bn1[i](con1[i](x)))
        # #     # x = torch.where(imgs > n14[i], 1., 0)
        # #     # imgs -= imgs * x
        # #     # # n = n / 2
        # #     # lis.append(bn1[i](con1[i](x)))
        # #     # if i==4:
        # #     #     lis.append(bn1[i](con1[i](imgsfu)))
        # #     # if i<4:
        # #     #     x = torch.where(imgs > n9[i], 1., 0)
        # #     #     imgs -= imgs * x
        # #     #     # n=n/2
        # #     #     lis.append(bn1[i](con1[i](x)))
        # #     # if i>4:
        # #     #     x = torch.where(imgs > n9[i-1], 1., 0)
        # #     #     imgs -= imgs * x
        # #     #     # n = n / 2
        # #     #     lis.append(bn1[i](con1[i](x)))
        # # lis=torch.stack(lis)
        # x=torch.cat([imgscopy.unsqueeze(0),lis],0).flatten(0, 1)

        # x = lis.flatten(0, 1)
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run

            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)



        return x
    def _profile_one_layer(self, m, x, dt):
        """Profiles a single layer's performance by computing GFLOPs, execution time, and parameters."""
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1e9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f"{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self):
        """Fuses Conv2d() and BatchNorm2d() layers in the model to improve inference speed."""
        LOGGER.info("Fusing layers... ")
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, "bn")  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):
        """Prints model information given verbosity and image size, e.g., `info(verbose=True, img_size=640)`."""
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        """Applies transformations like to(), cpu(), cuda(), half() to model tensors excluding parameters or registered
        buffers.
        """
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


class DetectionModel(BaseModel):
    # YOLOv5 detection model
    def __init__(self, cfg="yolov5s.yaml", ch=3, nc=None, anchors=None):
        """Initializes YOLOv5 model with configuration file, input channels, number of classes, and custom anchors."""
        super().__init__()

        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub

            self.yaml_file = Path(cfg).name
            with open(cfg, encoding="ascii", errors="ignore") as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override yaml value
        if anchors:
            LOGGER.info(f"Overriding model.yaml anchors with anchors={anchors}")
            self.yaml["anchors"] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(copy.deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml["nc"])]  # default names
        self.inplace = self.yaml.get("inplace", True)
        self.con1 = nn.ModuleList()
        for _ in range(timestep):
            self.con1.append(nn.Conv2d(3, 3, 3, 1, 1))

        print("timestep=", timestep)

        self.bn1 = nn.ModuleList()
        for _ in range(timestep):
            self.bn1.append(nn.BatchNorm2d(3))
        # self.con1=nn.Conv2d(3,3,3,1,padding=1)
        self.con2 = nn.Conv2d(3, 3, 3, 1,padding=1 )
        self.con3 = nn.Conv2d(3, 3, 3, 1, padding=1)
        self.bn3 = nn.BatchNorm2d(3)
        # self.bn1=nn.BatchNorm2d(3)
        self.bn2 = nn.BatchNorm2d(3)

        # self.mem=mem_update()
        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):

            def _forward(x):
                """Passes the input 'x' through the model and returns the processed output."""
                return self.forward(x)[0] if isinstance(m, Segment) else self.forward(x)

            s = 256  # 2x min stride
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])  # forward
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info("")


    def forward(self, x, augment=False, profile=False, visualize=False):
        """Performs single-scale or augmented inference and may include profiling or visualization."""
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize,self.con1,self.con2,self.bn1,self.bn2)  # single-scale inference, train

    def _forward_augment(self, x):
        """Performs augmented inference across different scales and flips, returning combined detections."""
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _descale_pred(self, p, flips, scale, img_size):
        """De-scales predictions from augmented inference, adjusting for flips and image size."""
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
        """Clips augmented inference tails for YOLOv5 models, affecting first and last tensors based on grid points and
        layer counts.
        """
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4**x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _initialize_biases(self, cf=None):
        """
        Initializes biases for YOLOv5's Detect() module, optionally using class frequencies (cf).

        For details see https://arxiv.org/abs/1708.02002 section 3.3.
        """
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5 : 5 + m.nc] += (
                math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())
            )  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


Model = DetectionModel  # retain YOLOv5 'Model' class for backwards compatibility


class SegmentationModel(DetectionModel):
    # YOLOv5 segmentation model
    def __init__(self, cfg="yolov5s-seg.yaml", ch=3, nc=None, anchors=None):
        """Initializes a YOLOv5 segmentation model with configurable params: cfg (str) for configuration, ch (int) for channels, nc (int) for num classes, anchors (list)."""
        super().__init__(cfg, ch, nc, anchors)


class ClassificationModel(BaseModel):
    # YOLOv5 classification model
    def __init__(self, cfg=None, model=None, nc=1000, cutoff=10):
        """Initializes YOLOv5 model with config file `cfg`, input channels `ch`, number of classes `nc`, and `cuttoff`
        index.
        """
        super().__init__()
        self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg)

    def _from_detection_model(self, model, nc=1000, cutoff=10):
        """Creates a classification model from a YOLOv5 detection model, slicing at `cutoff` and adding a classification
        layer.
        """
        if isinstance(model, DetectMultiBackend):
            model = model.model  # unwrap DetectMultiBackend
        model.model = model.model[:cutoff]  # backbone
        m = model.model[-1]  # last layer
        ch = m.conv.in_channels if hasattr(m, "conv") else m.cv1.conv.in_channels  # ch into module
        c = Classify(ch, nc)  # Classify()
        c.i, c.f, c.type = m.i, m.f, "models.common.Classify"  # index, from, type
        model.model[-1] = c  # replace
        self.model = model.model
        self.stride = model.stride
        self.save = []
        self.nc = nc

    def _from_yaml(self, cfg):
        """Creates a YOLOv5 classification model from a specified *.yaml configuration file."""
        self.model = None


def parse_model(d, ch):
    """Parses a YOLOv5 model from a dict `d`, configuring layers based on input channels `ch` and model architecture."""
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw, act, ch_mul = (
        d["anchors"],
        d["nc"],
        d["depth_multiple"],
        d["width_multiple"],
        d.get("activation"),
        d.get("channel_multiple"),
    )
    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        LOGGER.info(f"{colorstr('activation:')} {act}")  # print
    if not ch_mul:
        ch_mul = 8
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in {
            Conv,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            DWConv,
            MixConv2d,
            Focus,
            CrossConv,
            BottleneckCSP,
            C3,
            C3TR,
            C3SPP,
            C3Ghost,
            nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
        }:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, ch_mul)

            args = [c1, c2, *args[1:]]
            if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x}:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        # TODO: channel, gw, gd
        elif m in {Detect, Segment}:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            if m is Segment:
                args[3] = make_divisible(args[3] * gw, ch_mul)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f"{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}")  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="yolov5s.yaml", help="model.yaml")
    parser.add_argument("--batch-size", type=int, default=1, help="total batch size for all GPUs")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--profile", action="store_true", help="profile model speed")
    parser.add_argument("--line-profile", action="store_true", help="profile model speed layer by layer")
    parser.add_argument("--test", action="store_true", help="test all yolo*.yaml")
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))
    device = select_device(opt.device)

    # Create model
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    model = Model(opt.cfg).to(device)

    # Options
    if opt.line_profile:  # profile layer by layer
        model(im, profile=True)

    elif opt.profile:  # profile forward-backward
        results = profile(input=im, ops=[model], n=3)

    elif opt.test:  # test all models
        for cfg in Path(ROOT / "models").rglob("yolo*.yaml"):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f"Error in {cfg}: {e}")

    else:  # report fused model summary
        model.fuse()
    # def _forward_once(self, imgs, profile=False, visualize=False,con1=None,con2=None,bn1=None,bn2=None):
    #     """Performs a forward pass on the YOLOv5 model, enabling profiling and feature visualization options."""
    #     y, dt = [], []  # outputs
    #     # x=imgs
    #     # x = bn2(con2(imgs))
    #
    #     x=imgs.unsqueeze(0).repeat(timestep,1,1,1,1).flatten(0, 1)
    #     # x = self.conv1_s(x)
    #
    #
    #
    #     # #
    #     # sigmoid = nn.Sigmoid()
    #     #
    #     #
    #     # lano = nn.LayerNorm(normalized_shape=[3, imgs.size(2), imgs.size(3)]).to(imgs.device)
    #     # # imgsfu=imgs.clone()
    #     # imgscopy=bn1[timestep-1](con1[timestep-1](imgs))
    #     # # n31 = [0.8662419354838822, 0.8233838709677542, 0.7900258064516253, 0.760667741935496, 0.7340096774193664,
    #     # #        0.7097516129032364, 0.6863935483871064, 0.6642354838709763, 0.6427774193548461, 0.6217193548387158,
    #     # #        0.6015612903225854, 0.5815032258064551, 0.5617451612903247, 0.5421870967741942, 0.5212290322580639,
    #     # #        0.5004709677419337, 0.47951290322580337, 0.4585548387096731, 0.4369967741935429, 0.4153387096774127,
    #     # #        0.39318064516128254, 0.37072258064515246, 0.34786451612902236, 0.3241064516128924, 0.2995483870967625,
    #     # #        0.27339032258063284, 0.24513225806450334, 0.2140741935483781, 0.17791612903225396, 0.1303580645161311,
    #     # #        0.0]
    #     # # n19 = [0.8370684210526437, 0.7810368421052757, 0.7353052631579065, 0.6955736842105367, 0.6590421052631664,
    #     # #         0.6244105263157961, 0.5910789473684255, 0.558147368421055, 0.5254157894736843, 0.49188421052631376,
    #     # #         0.45815263157894326, 0.42432105263157277, 0.38908947368420244, 0.35325789473683217, 0.3148263157894622,
    #     # #         0.2736947368420925, 0.22716315789472616, 0.17013157894736552, 0.0]
    #     # n14 = [0.8145714285714412, 0.7472428571428692, 0.6929142857142959, 0.6429857142857222, 0.596157142857148,
    #     #               0.5513285714285736, 0.5053999999999994, 0.4598714285714251, 0.41354285714285094, 0.3656142857142769,
    #     #               0.3142857142857033, 0.2575571428571302, 0.18842857142856598, 0.0]
    #     # # n9 = [0.7743888888889014, 0.6856777777777879, 0.6101666666666729, 0.5398555555555573, 0.46854444444444177,
    #     # #        0.39583333333332643, 0.31822222222221164, 0.22701111111110026, 0.0]
    #     # #
    #     # # one = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    #     # # two = [1, 2, 5, 6, 9, 10, 13, 14, 17, 18, 21, 22, 25, 26, 29, 30]
    #     # # three = [3, 4, 5, 6, 11, 12, 13, 14, 19, 20, 21, 22, 27, 28, 29, 30]
    #     # # four = [7, 8, 9, 10, 11, 12, 13, 14, 23, 24, 25, 26, 27, 28, 29, 30]
    #     # # five = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    #     # lis = []
    #     # # lis = torch.zeros(timestep - 1, imgs.size(0), 3, imgs.size(2), imgs.size(3))
    #     # # device = imgs.device
    #     # # lis = lis.to(device)
    #     # imgs = lano(imgs)
    #     # imgs = sigmoid(imgs)
    #     # # n=0.5
    #     # for i in range(14):
    #     #
    #     #     # z = (i + 1.0)
    #     #     # z = 1. - z / (timestep-1)
    #     #     # z = z.__float__()
    #     #     x = torch.where(imgs > n14[i], 1., 0)
    #     #     imgs -= imgs * x
    #     #     # if i in one:
    #     #     #     lis[0] += x
    #     #     #
    #     #     # if i in two:
    #     #     #     lis[1] += x
    #     #     #
    #     #     # if i in three:
    #     #     #     lis[2] += x
    #     #     #
    #     #     # if i in four:
    #     #     #     lis[3] += x
    #     #     #
    #     #     # if i in five:
    #     #     #     lis[4] += x
    #     #     # imgs -= n * x
    #     #     # n=n/2
    #     #     lis.append(bn1[i](con1[i](x)))
    #     #     # x = torch.where(imgs > n14[i], 1., 0)
    #     #     # imgs -= imgs * x
    #     #     # # n = n / 2
    #     #     # lis.append(bn1[i](con1[i](x)))
    #     #     # if i==4:
    #     #     #     lis.append(bn1[i](con1[i](imgsfu)))
    #     #     # if i<4:
    #     #     #     x = torch.where(imgs > n9[i], 1., 0)
    #     #     #     imgs -= imgs * x
    #     #     #     # n=n/2
    #     #     #     lis.append(bn1[i](con1[i](x)))
    #     #     # if i>4:
    #     #     #     x = torch.where(imgs > n9[i-1], 1., 0)
    #     #     #     imgs -= imgs * x
    #     #     #     # n = n / 2
    #     #     #     lis.append(bn1[i](con1[i](x)))
    #     # lis=torch.stack(lis)
    #     # x=torch.cat([imgscopy.unsqueeze(0),lis],0).flatten(0, 1)
