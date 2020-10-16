from argparse import ArgumentParser
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import pytorch_lightning as pl

from networks import (
    Encoder,
    GlobalMIDiscriminator,
    LocalMIDiscriminator,
    DistributionDiscriminator)

from criteria import (JensenShannonMIEstimator, KLDivergenceEstimator)

from utils import timestamp

ngpu = 1

alpha = 0.1
beta = 1
gamma = 0.1

batch_size = 16

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

trainset = datasets.CIFAR10(root='./data',
                            train=True,
                            download=True,
                            transform=transform)

trainloader = DataLoader(trainset,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=2)
 
testset = datasets.CIFAR10(root='./data',
                           train=False,
                           download=True,
                           transform=transform)

testloader = DataLoader(testset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
           'ship', 'truck')


class DeepInfoMax(pl.LightningModule):
    def __init__(self, ngpu: int):
        super().__init__()
        self.E = Encoder(ngpu)
        self.GMID = GlobalMIDiscriminator(ngpu)
        self.LMID = LocalMIDiscriminator(ngpu)
        self.DD = DistributionDiscriminator(ngpu)

        self.LMI_estimator = JensenShannonMIEstimator(self.E, self.LMID)
        self.GMI_estimator = JensenShannonMIEstimator(self.E, self.GMID)
        self.KLD_estimator = KLDivergenceEstimator(self.DD)

    def foward(self, x):
        return self.E(x)

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, _ = batch

        result = None
        if optimizer_idx == 0:
            result = self.minimizer_step(x)

        if optimizer_idx == 1:
            result = self.maximizer_step(x)

        return result

    def minimizer_step(self, x):
        """ Argmin KLD term, prior matching."""
        return gamma * self._kld

    def _kld(self, x):
        """ Sampled from uniform distiribution. """
        _, global_features  = self.E(x)
        sampled_features = torch.rand(global_features.shape, device=self.device)
        KLD = self.KLD_estimator(global_features, sampled_features)
        return KLD
    
    def maximizer_step(self, x):
        """ Argmax all terms. """
        real, fake = self._make_real_fake_batches(x)
        globalMI = GMI_estimator(real, fake)
        localMI = LMI_estimator(real, fake)
        KLD = self._kld(x)

        neg_objective = -(alpha * globalMI + beta * localMI + gamma * KLD)

        self.log('minimizing neg_objective:', neg_objective, on_epoch=True, prog_bar=True)
        return neg_objective

    def _make_real_fake_batches(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        for i in range(self.batch_size):
            # the others of inputs[i]
            fake_ind = torch.tensor(True).repeat(self.batch_size)
            fake_ind[i] = False
            if i == 0:
                real = x[i].repeat(self.batch_size - 1, 1, 1, 1)
                fake = x[fake_ind].clone()
            real = torch.cat((real, x[i].repeat(self.batch_size - 1, 1, 1, 1)))
            fake = torch.cat((fake, x[fake_ind].clone()))
        return real, fake

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        m = self.hparams.sgd_momentum

        params = (list(self.E.parameters()) + list(self.GMID.parameters()) 
                + list(self.MID.parameters()) + list(self.DD.parameters()))
        maximizer = optim.SGD(params, lr=lr, momentum=m)
        minimizer = optim.SGD(self.E.parameters(), lr=lr, momentum=m)
        return [minimizer, maximizer],  []



def main(hparams):
    model = DeepInfoMax()
    trainer = Trainer(gpus=hparams.gpus)
    trainer.fit(model)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpus', default=None)
    args = parser.parse_args()

    main(args)