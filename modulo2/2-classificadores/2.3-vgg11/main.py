'''
Curso sobre Visão Computacional
Prof. Adriano Santos, PhD
'''
# VGG11 https://iq.opengenus.org/vgg-11/
# Artigo: 

import torchvision.transforms as transforms
import torchvision
import torch
from torch import nn, optim
import argparse
import logging
from PIL import Image
import numpy as np
from model import VGG11

from util import Preprocessador, getLabel

# Funcao de treino
def main(args=None):
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    logging.info('Arquitetura VGG11')

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', help='Num. de classes.', type=int, default=1000)
    parser.add_argument('--pretrained', help='Serão utilizados pesos pré-treinados.', type=bool, default=True)
    parser.add_argument('--model_url', help='Caminho para os pesos.', default="./pesos/vgg11-bbd30ac9.pth")
    
    opt = parser.parse_args(args)

    # Dados
    proc = Preprocessador()
    imagem_url = "./imagens/raposa.jpg"
    imagem = Image.open(imagem_url)
    
    imagem = proc.executa(imagem)
    imagem = imagem.unsqueeze(0)

    
    # Instancia do modelo
    model = VGG11(opt.num_classes)
    model.eval()

    # Caso deseje utilizar os pesos pré-treinados
    if opt.pretrained:
        checkpoint = torch.load(opt.model_url)
        model.load_state_dict(checkpoint)

    # Utiliza a GPU se existir no computador
    if torch.cuda.is_available():
        model.to('cuda')

    with torch.no_grad():
        saida = model(imagem)

    # Obtem o indice melhor ranqueado
    index = np.argmax(saida[0]).item()
    acuracia = torch.max(saida).item()
    
    print(getLabel(index), acuracia)

if __name__ == "__main__":
    main()