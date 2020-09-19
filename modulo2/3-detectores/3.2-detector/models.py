# Estrutura básica para projetos de Machine Learning e Deep Learning
# Por Adriano Santos.

from torch import nn, relu
import torch.nn.functional as F
import torch.optim as optim
import torch
from torchvision import models


class ResNet(nn.Module):
    def __init__(self, saida, pretreinado=True):
        super(ResNet, self).__init__()

        resnet = models.resnet34(pretrained=pretreinado)
        layers = list(resnet.children())[:8]

        self.features1 = nn.Sequential(*layers[:6])
        self.features2 = nn.Sequential(*layers[6:])

        self.classificador = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, saida))

    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = F.relu(x)
        x = nn.AdaptiveAvgPool2d((1,1))(x)
        x = x.view(x.shape[0], -1)
        return self.classificador(x)