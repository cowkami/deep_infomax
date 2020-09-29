import torch

from networks import Encoder, LocalMIDiscriminator, DistributionDiscriminator


class JensenShannonMIEstimator:
    """Jensen Shannon Divergence Mutual Infomation Estimator"""
    def __init__(self, encoder: Encoder, discriminator: LocalMIDiscriminator):
        self.E = encoder
        self.T = discriminator

    def __call__(self, real_inputs, fake_inputs) -> torch.Tensor:
        def sp(z) -> torch.Tensor:
            return torch.log(1 + torch.exp(z))

        real_feature_map = self.E.conv_net(real_inputs)
        fake_feature_map = self.E.conv_net(fake_inputs)
        global_feature = self.E.fc_net(real_feature_map)

        Ep = torch.mean(-sp(-self.T(real_feature_map, global_feature)))
        Epp = torch.mean(sp(self.T(fake_feature_map, global_feature)))
        return Ep - Epp


class KLDivergenceEstimator:
    """Kullbuck Leibler Divergence Estimator.
    
    Calculate the divergence between 
    the push-forward distiribution and a prior. 
    """
    def __init__(self, discriminator: DistributionDiscriminator):
        self.D = discriminator

    def __call__(self, global_features: torch.Tensor,
                 sampled_features: torch.Tensor) -> torch.Tensor:
        Ev = torch.mean(torch.log(self.D(sampled_features)))
        Ep = torch.mean(torch.log(1 - self.D(global_features)))
        return Ev + Ep