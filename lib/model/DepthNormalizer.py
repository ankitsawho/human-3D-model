import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthNormalizer(nn.Module):
    def __init__(self, opt):
        super(DepthNormalizer, self).__init__()
        self.opt = opt

    def forward(self, z, calibs=None, index_feat=None):
        z_feat = z * (self.opt.loadSize // 2) / self.opt.z_size
        return z_feat
