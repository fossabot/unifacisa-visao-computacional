# Estrutura básida para projetos de Machine Learning e Deep Learning
# Por Adriano Santos.

import torch
from torch import nn, optim
import torchvision.transforms as transforms
from models import ResNet
from torch.autograd import Variable
import util
from dataset import DataSet


# Modulos para auxilio na estrutura do projeto.
from tqdm import tqdm
import argparse
import logging
import numpy as np
from PIL import Image 
import cv2
import time
import random

# Garante a replicabilidade
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def main():
    # ************************************ DADOS ***************************************************
    padroniza_imagem = 300
    tamanho_da_entrada = (224, 224)
    arquivo = "./imagens/raposa.jpg"
    cor = (0, 255, 0)
    
    # Operacoes de preprocessamento e augumentacao
    composicao_de_transformacao = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),])

    # ************************************* REDE ************************************************
    modelo = ResNet(1000, True)
    modelo.eval()

    # Abre a imagem
    imagem_original = np.asarray(Image.open(arquivo))
    imagem = imagem_original.copy()


    # Obtem as coordenadas da imagem
    (H, W) = imagem.shape[:2]
    r = padroniza_imagem / W
    dim_final = (padroniza_imagem, int(H * r))
    imagem = cv2.resize(imagem, dim_final, interpolation = cv2.INTER_AREA)

    # Area da regiao de interesse
    ROI = (150, 150) #(H,W)
    
    # Lista de regioes de interesse (rois) e coods (coordenadas) 
    rois = []
    coods = []
      
    # Execucao da funcao de piramede
    for nova_imagem in util.image_pyramid(imagem, escala=1.2): 
        # Fator de escala entre a imagem original e a nova imagem gerada
        fator_escalar = W / float(nova_imagem.shape[1])

        # Executa a operacao de deslizamento de janela
        for (x, y, regiao) in util.sliding_window(nova_imagem, size=ROI, stride=8):
            
            # Condicao de parada
            key = cv2.waitKey(1) & 0xFF
            if (key == ord("q")):
                break
            
            if regiao.shape[0] != ROI[0] or regiao.shape[1] != ROI[1]:
                continue
            
            # Obtem as coordenadas da ROI com relacao aa image
            x_r = int(x * fator_escalar)
            w_r = int(fator_escalar * ROI[1])

            y_r = int(y * fator_escalar)
            h_r = int(fator_escalar * ROI[0])

            # Obtem o ROI e realiza a transformacao necessaria para o treinamento
            roi = cv2.resize(regiao, tamanho_da_entrada)
            roi = np.asarray(roi)
            rois.append(roi)

            # Obtem as coordenadas (x1, y1, x2, y2)
            coods.append((x_r, y_r, x_r + w_r, y_r + h_r))

            # Utiliza uma copia da imagem
            copia = nova_imagem.copy()
            # Imprime um retangulo na imagem de acordo com a posicao
            cv2.rectangle(copia, (x, y), (x + ROI[1], y + ROI[0]), cor, 2)
            # Mostra o resultado na janela
            cv2.imshow("Janela", copia[:, :, ::-1])
            
            # Atraso no loop
            time.sleep(0.01)

    # Fechar todas as janelas abertas
    cv2.destroyAllWindows()

    #rois = np.array(rois, dtype="float32") # transform to torch tensor
    dataset = DataSet(rois, coods, composicao_de_transformacao)
    size = len(dataset)
    train_loader = torch.utils.data.DataLoader(dataset=dataset, shuffle=True, batch_size=size)

    print("Cópias: ", size)
    with torch.no_grad():
        for _, (X, y) in enumerate(train_loader):
            # Classificacoes de todas as copias das imagens
            resultado = modelo.forward(X)
            
            # Obtem os melhores resultados por imagem
            confs, indices_dos_melhores_resultados = torch.max(resultado,1)
            classe, _ = torch.mode(indices_dos_melhores_resultados.flatten(),-1)

            # Mascara
            mascara = [True if item == classe else False for item in indices_dos_melhores_resultados ]

            # Selecao de boxes
            boxes = []
            for i in range(size):
                if mascara[i] == True:
                    boxes.append(coods[i])

            # Realiza operacao de non_max_suppression
            boxes = util.non_max_suppression(np.asarray(boxes), overlapThresh=0.3)

            copia = imagem_original.copy()
            for (x1, y1, x2, y2) in boxes:
                cv2.rectangle(copia, (x1, y1), (x2, y2),cor, 2)
                
            cv2.imshow("Final", copia[:, :, ::-1])
            cv2.waitKey(0) 

    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Main function.
    main()

