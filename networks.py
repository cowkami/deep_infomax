import torch
import torch.nn as nn
import torch.nn.functional as F

nc = 3
ndf = 64
dim_feature = 64


class Encoder(nn.Module):
    """DGI Encoder
    
    Based on DCGAN's Discriminator
    """
    def __init__(self, ngpu):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        self.C = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True))
        # state size. (ndf*2) x 8 x 8

        self.f = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # output size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, ndf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 1 x 1
            nn.Flatten(),
            # state size. (ndf*4)
            nn.Linear(ndf * 4, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 1024
            nn.Linear(1024, dim_feature))
        # output size. 64

    def forward(self, x) -> torch.Tensor:
        MxM_features = self.C(x)
        global_feature = self.f(MxM_features)
        return MxM_features, global_feature


class LocalMIDiscriminator(nn.Module):
    """Local Mutual Information Discriminator.
    
    Concat-and-convolve architecture.
    """
    def __init__(self, ngpu):
        super(LocalMIDiscriminator, self).__init__()
        self.ngpu = ngpu
        self.net = nn.Sequential(
            nn.Conv2d(dim_feature + ndf * 4, 512, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, 1),  # 1 x 1 conv 
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),  # 1 x 1 conv
            nn.Conv2d(512, 1, 1))

    def forward(self, local_feature_map, global_feature) -> torch.Tensor:
        n, c, h, w = local_feature_map.shape
        global_feature_map = (
            torch.ones((n, dim_feature, h, w)).to(local_feature_map.device) *
            global_feature.reshape(*global_feature.shape, 1, 1))
        global_local_feature = torch.cat(
            (local_feature_map, global_feature_map), dim=1)
        return self.net(global_local_feature)


class DistributionDiscriminator(nn.Module):
    """Discriminator whether global feature from a prior"""
    def __init__(self, ngpu):
        super(DistributionDiscriminator, self).__init__()
        self.ngpu = ngpu
        self.net = nn.Sequential(nn.Linear(dim_feature, 1000),
                                 nn.LeakyReLU(0.2, inplace=True),
                                 nn.Linear(1000, 200),
                                 nn.LeakyReLU(0.2, inplace=True),
                                 nn.Linear(200, 1), nn.Sigmoid())

    def forward(self, global_features) -> torch.Tensor:
        return self.net(global_features)