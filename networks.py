from typing import Tuple

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
    def __init__(self, ngpu: int):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        self.conv = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True))
        # state size. (ndf*2) x 8 x 8

        self.fc = nn.Sequential(
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_feature_map = self.conv(x)
        global_feature = self.fc(local_feature_map)
        return local_feature_map, global_feature


class MIDiscriminator(nn.Module):
    """Base class of global and local mutual information discriminator."""
    def __init__(self, ngpu: int):
        super(MIDiscriminator, self).__init__()
        self.ngpu = ngpu
        self.net = None
    
    def forward(self,
                local_feature_map: torch.Tensor,
                global_feature: torch.Tensor) -> torch.Tensor:
        L, G = self._preprocess(local_feature_map, global_feature)
        return self.net(torch.cat((L, G), dim=1))

    def _preprocess(self,
            l_map: torch.Tensor,
            g_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return None, None


class GlobalMIDiscriminator(MIDiscriminator):
    """Global Mutual Information Discriminator."""
    def __init__(self, ngpu: int):
        super(GlobalMIDiscriminator, self).__init__(ngpu)
        self.net = nn.Sequential(nn.Linear(dim_feature + ndf * 2 * 8 * 8, 512),
                                 nn.LeakyReLU(0.2, inplace=True),
                                 nn.Linear(512, 512),
                                 nn.LeakyReLU(0.2, inplace=True),
                                 nn.Linear(512, 1))
    
    def _preprocess(self,
            l_map: torch.Tensor, 
            g_vec: torch.Tensor) -> torch.Tensor:
        l_vec = l_map.view(-1, ndf * 2 * 8 * 8)
        return l_vec, g_vec


class LocalMIDiscriminator(MIDiscriminator):
    """Local Mutual Information Discriminator.
    
    Concat-and-convolve architecture.
    """
    def __init__(self, ngpu: int):
        super(LocalMIDiscriminator, self).__init__(ngpu)
        self.net = nn.Sequential(
            nn.Conv2d(dim_feature + ndf * 2, 512, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, 1),  # 1 x 1 conv 
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),  # 1 x 1 conv
            nn.Conv2d(512, 1, 1))

    def _preprocess(self,
            l_map: torch.Tensor, 
            g_vec: torch.Tensor) -> torch.Tensor:
        n, c, h, w = l_map.shape
        g_map = (
            torch.ones((n, dim_feature, h, w), device=g_vec.device)
            * g_vec.reshape(*g_vec.shape, 1, 1))
        return l_map, g_map


class DistributionDiscriminator(nn.Module):
    """Discriminator whether global feature from a prior"""
    def __init__(self, ngpu: int):
        super(DistributionDiscriminator, self).__init__()
        self.ngpu = ngpu
        self.net = nn.Sequential(nn.Linear(dim_feature, 1000),
                                 nn.LeakyReLU(0.2, inplace=True),
                                 nn.Linear(1000, 200),
                                 nn.LeakyReLU(0.2, inplace=True),
                                 nn.Linear(200, 1),
                                 nn.Sigmoid())

    def forward(self, global_feature: torch.Tensor) -> torch.Tensor:
        return self.net(global_feature)