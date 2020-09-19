
import torch
from torchvision import models

resnet = models.resnet34(pretrained=False)

print(resnet)

'''
layers = list(resnet.children())

print(len(layers))

for layer in layers[:5]:
    print(layer)
'''