import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasePIFuNet import BasePIFuNet
from .SurfaceClassifier import SurfaceClassifier
from .DepthNormalizer import DepthNormalizer
from .ConvFilters import *
from ..net_util import init_net

class ConvPIFuNet(BasePIFuNet):
    def __init__(self,
                 opt,
                 projection_mode='orthogonal',
                 error_term=nn.MSELoss(),
                 ):
        super(ConvPIFuNet, self).__init__(
            projection_mode=projection_mode,
            error_term=error_term)

        self.name = 'convpifu'

        self.opt = opt
        self.num_views = self.opt.num_views

        self.image_filter = self.define_imagefilter(opt)

        self.surface_classifier = SurfaceClassifier(
            filter_channels=self.opt.mlp_dim,
            num_views=self.opt.num_views,
            no_residual=self.opt.no_residual,
            last_op=nn.Sigmoid())

        self.normalizer = DepthNormalizer(opt)

        self.im_feat_list = []

        init_net(self)

    def define_imagefilter(self, opt):
        net = None
        if opt.netIMF == 'multiconv':
            net = MultiConv(opt.enc_dim)
        elif 'resnet' in opt.netIMF:
            net = ResNet(model=opt.netIMF)
        elif opt.netIMF == 'vgg16':
            net = Vgg16()
        else:
            raise NotImplementedError('model name [%s] is not recognized' % opt.imf_type)

        return net

    def filter(self, images):
        '''
        Filter the input images
        store all intermediate features.
        :param images: [B, C, H, W] input images
        '''
        self.im_feat_list = self.image_filter(images)

    def query(self, points, calibs, transforms=None, labels=None):
        if labels is not None:
            self.labels = labels

        xyz = self.projection(points, calibs, transforms)
        xy = xyz[:, :2, :]
        z = xyz[:, 2:3, :]

        z_feat = self.normalizer(z)

        # This is a list of [B, Feat_i, N] features
        point_local_feat_list = [self.index(im_feat, xy) for im_feat in self.im_feat_list]
        point_local_feat_list.append(z_feat)
        # [B, Feat_all, N]
        point_local_feat = torch.cat(point_local_feat_list, 1)

        self.preds = self.surface_classifier(point_local_feat)
