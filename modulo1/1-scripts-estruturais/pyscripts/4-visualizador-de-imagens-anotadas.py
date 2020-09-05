# ****************************************************************
# AULA: Visão Computacional
# Prof: Adriano A. Santos, DSc.
# ****************************************************************
import sys
sys.path.append('../')
import util

import cv2
import os
import numpy as np
import random

# Obtem imagem
img_path = "../../dataset/raposa.jpg"
img = cv2.imread(img_path)

# Define cor para criacao das caixas
color = (255, 0, 0) 

# Obtem altura e largura da imagem
(H, W) = img.shape[:2]

# Obtem anotacoes
lines = util.obtemLinhas(img_path.replace(".jpg", ".txt"))
annots = [item for item in lines]

# Para cada anotacao, crie uma caixa
# TODO: Toda a logica de conversao deve ficar no modulo util.

for i, annt in enumerate(annots):
    # Obtem coordenadas da caixa
    box = annt.split()

    # Obtem categoria
    cat = int(box[0])

    # Obtem as dimensoes w, h, x, y de cada caixa
    w = float(box[3])
    h = float(box[4])
    x = float(box[1]) 
    y = float(box[2])

    # Obtem os pontos iniciais
    x = x - w / 2
    y = y - h / 2

    # Ajusta as metricas de acordo com o valor da altura e largura da imagem
    x = int(x * W)
    w = int(w * W)
    y = int(y * H)
    h = int(h * H)

    # Conversao final das medidas w,h,x,y para x1,x2,y1,y2
    x1, x2, y1, y2 = [x, x+w, y, y+h]
    
    # Desenha o retangulo
    cv2.rectangle(img,(x1, y1), (x2, y2), color ,2)

cv2.imshow('Imagem anotada',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


