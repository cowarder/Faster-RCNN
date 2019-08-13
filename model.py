from torchvision.ops import RoIPool
import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self):
        pass

    def roi_pooling(self, out_h, out_w, scale):
        return RoIPool((out_h, out_w), scale)