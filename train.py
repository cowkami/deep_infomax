from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from networks import (
    Encoder, 
    LocalMIDiscriminator, 
    DistributionDiscriminator)

from criteria import (
    JensenShannonMIEstimator,
    KLDivergenceEstimator)

from utils import timestamp


ngpu = 1

beta = 1
gamma = 0.1

batch_size = 16

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# define networks
netE = Encoder(ngpu)
netE.to(device)

netLMID = LocalMIDiscriminator(ngpu)
netLMID.to(device)

netDD = DistributionDiscriminator(ngpu)
netDD.to(device)

# define criteria
MI_estimator = JensenShannonMIEstimator(netE, netLMID)
KLD_estimator = KLDivergenceEstimator(netDD)

params = (
    list(netE.parameters())
    + list(netLMID.parameters()) 
    + list(netDD.parameters()))
maximizer = optim.SGD(params, lr=0.001, momentum=0.9)
minimizer = optim.SGD(netE.parameters(), lr=0.001, momentum=0.9)

def make_real_fake_batches(inputs) -> Tuple[torch.Tensor, torch.Tensor]:
    for i in range(batch_size):
        # the others of inputs[i]
        fake_ind = torch.tensor(True).repeat(batch_size)
        fake_ind[i] = False
        if i == 0:
            real = inputs[i].repeat(batch_size - 1, 1, 1, 1)
            fake = inputs[fake_ind].clone()
        real = torch.cat((real, inputs[i].repeat(batch_size - 1, 1, 1, 1)))
        fake = torch.cat((fake, inputs[fake_ind].clone()))
    return real, fake

for epoch in range(100):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)


        ##################################
        # argmin KLD term, prior matching.
        ##################################
        netE.zero_grad()

        global_features = netE(inputs)
        # sampled from uniform distiribution
        sampled_features = torch.rand(global_features.shape).to(device)  
        KLD = gamma * KLD_estimator(global_features, sampled_features)
        KLD.backward()
        minimizer.step()

        ###################
        # argmax all terms.
        ###################
        netE.zero_grad()
        netDD.zero_grad()
        netLMID.zero_grad()

        real, fake = make_real_fake_batches(inputs)
        LocalMI = MI_estimator(real, fake)

        global_features = netE(inputs)
        sampled_features = torch.rand(global_features.shape).to(device)  
        KLD = KLD_estimator(global_features, sampled_features)
        neg_objective = - (beta * LocalMI + gamma * KLD)
        neg_objective.backward()
        maximizer.step()

        running_loss += neg_objective.item()
        if i % 1000 == 999:
            print(f'[{epoch + 1: d}, {i + 1: 5d}] negative objective : {running_loss / 1000}')
            running_loss = 0.0
print('Finished Training')

#except KeyboardInterrupt:
#    print('\nstop training')

#torch.save(encoder.state_dict(), f'./data/weights/{timestamp()}.pth')