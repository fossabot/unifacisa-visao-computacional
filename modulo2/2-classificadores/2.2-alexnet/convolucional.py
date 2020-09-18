import torch
import torch.nn as nn
from PIL import Image
from util import Preprocessador
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

#m = nn.AdaptiveAvgPool2d((3,3))
#maxi = nn.MaxPool2d((5,7))

imagem_url = "./imagens/raposa.jpg"
    
imagem = Image.open(imagem_url)
imagem.show()

imagem= imagem.convert('L')
imagem.show()

proc = Preprocessador()
imagem = proc.executa(imagem)
imagem = imagem.unsqueeze(0)

# Conv 1
conv1 = nn.Conv2d(1, 1, kernel_size=11, stride=4, padding=2)
re = nn.ReLU(conv1)
m = nn.MaxPool2d(kernel_size=3, stride=2)

saida_cv1 = conv1(imagem)
saida = re(saida_cv1)
saida = m(saida)

kernels = saida.detach()
fig, axarr = plt.subplots(kernels.size(0))
plt.imshow(kernels[0].squeeze())
plt.show()

# conv 2
conv2 = nn.Conv2d(1, 50, kernel_size=5, padding=2)
re = nn.ReLU(conv2)
m = nn.MaxPool2d(kernel_size=3, stride=2)

saida_cv2 = conv2(saida)
saida = re(saida_cv2)
saida = m(saida)

kernels = saida.detach().clone()
imagens = kernels[0].squeeze()

linhas, colunas = int(len(imagens)/10), int(len(imagens)/5)

fig, axarr = plt.subplots(linhas, colunas,
                       sharex='col', 
                       sharey='row')

for linha in range(linhas):
    for coluna in range(colunas):
        axarr[linha, coluna].imshow(imagens[linha+coluna].numpy())

plt.show()


# conv 3
conv3 = nn.Conv2d(50, 50, kernel_size=5, padding=2)
re = nn.ReLU(conv2)
m = nn.MaxPool2d(kernel_size=3, stride=2)

saida_cv3 = conv3(saida)
saida = re(saida_cv3)
saida = m(saida)

kernels = saida.detach().clone()
imagens = kernels[0].squeeze()

linhas, colunas = int(len(imagens)/10), int(len(imagens)/5)

fig, axarr = plt.subplots(linhas, colunas,
                       sharex='col', 
                       sharey='row')

for linha in range(linhas):
    for coluna in range(colunas):
        axarr[linha, coluna].imshow(imagens[linha+coluna].numpy())

plt.show()