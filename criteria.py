import torch
import torch.nn as nn

from networks import Encoder, MIDiscriminator, DistributionDiscriminator


class JensenShannonMIEstimator(nn.Module):
    """Jensen Shannon Divergence Mutual Infomation Estimator"""
    def __init__(self, encoder: Encoder, discriminator: MIDiscriminator):
        super(JensenShannonMIEstimator, self).__init__()
        self.E = encoder
        self.T = discriminator

    def forward(self, real_input, fake_input) -> torch.Tensor:
        real_feature_map = self.E.conv(real_input)
        fake_feature_map = self.E.conv(fake_input)
        global_feature = self.E.fc(real_feature_map)

        sp = torch.nn.Softplus()
        Ep = -sp(-self.T(real_feature_map, global_feature)).mean()
        Epp = sp(self.T(fake_feature_map, global_feature)).mean()
        return Ep - Epp


class KLDivergenceEstimator(nn.Module):
    """Kullbuck Leibler Divergence Estimator.
    
    Calculate the divergence between 
    the push-forward distiribution and a prior. 
    """
    def __init__(self, discriminator: DistributionDiscriminator):
        super(KLDivergenceEstimator, self).__init__()
        self.D = discriminator

    def forward(self, global_feature: torch.Tensor, sampled_feature: torch.Tensor) -> torch.Tensor:
        Ev = torch.mean(torch.log(self.D(sampled_feature)))
        Ep = torch.mean(torch.log(1 - self.D(global_feature)))
        return Ev + Ep