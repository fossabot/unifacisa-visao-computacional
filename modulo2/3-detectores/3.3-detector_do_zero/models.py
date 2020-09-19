# Estrutura básica para projetos de Machine Learning e Deep Learning
# Por Adriano Santos.

from torch import nn, relu
import torch.nn.functional as F
import torch.optim as optim
import torch
from torchvision import models

class RedeComum(nn.Module):
    def __init__(self, entrada, hidden_1, hidden_2, saida):
        super(RedeComum, self).__init__()

        self.classificador = nn.Sequential(
                                nn.Linear(entrada, hidden_1),
                                nn.ReLU(inplace=True),
                                nn.Linear(hidden_1, hidden_2),
                                nn.ReLU(inplace=True),
                                nn.Linear(hidden_2, hidden_2),
                                nn.ReLU(inplace=True),
                                nn.Linear(hidden_2, saida),
        )

    def forward(self, x):
        return self.classificador(x)


class CNN(nn.Module):
    def __init__(self, hidden_1, hidden_2, saida):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
                        nn.Conv2d(3, 32, kernel_size=3),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(3, stride=1, padding=0),
                        nn.Conv2d(32, 64, kernel_size=3),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(3, stride=1, padding=0),
                        nn.Conv2d(64, 16, kernel_size=3),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(3, stride=1, padding=0)
        )
        self.classificador = nn.Sequential(
                        nn.Linear(719104, hidden_1),
                        nn.ReLU(inplace=True),
                        nn.Linear(hidden_1, hidden_2),
                        nn.ReLU(inplace=True),
                        nn.Linear(hidden_2, hidden_2),
                        nn.ReLU(inplace=True),
                        nn.Linear(hidden_2, saida),
        )
    

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.shape[0], -1)
        return self.classificador(out)


class AlexNet(nn.Module):

    def __init__(self, hidden_1, hidden_2, output_size):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # https://www.programmersought.com/article/73681181623/
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.entry = nn.Linear(256 * 6 * 6, hidden_1)
        self.hidden = nn.Linear(hidden_1, hidden_2)
        self.hidden2 = nn.Linear(hidden_2, hidden_2)
        self.out = nn.Linear(hidden_2, output_size)
        

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = relu(self.entry(out))
        out = relu(self.hidden(out))
        out = relu(self.hidden2(out))
        return self.out(out)

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


class AdrianoNet(nn.Module):
    def __init__(self, saida, pretreinado=True):
        super(AdrianoNet, self).__init__()

        resnet = models.resnet34(pretrained=pretreinado)
        layers = list(resnet.children())[:8]

        self.extrator1 = nn.Sequential(*layers[:6])
        self.extrator2 = nn.Sequential(*layers[6:])

        self.classificador = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, saida))
        self.regressor = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 4))

    def forward(self, x):
        x = self.extrator1(x)
        x = self.extrator2(x)
        x = F.relu(x)
        x = nn.AdaptiveAvgPool2d((1,1))(x)
        x = x.view(x.shape[0], -1)
        return self.classificador(x), self.regressor(x) 

