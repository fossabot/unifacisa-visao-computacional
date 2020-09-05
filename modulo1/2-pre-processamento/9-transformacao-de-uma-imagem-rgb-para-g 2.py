# ****************************************************************
# AULA: Visão Computacional
# Prof: Adriano A. Santos, DSc.
# ****************************************************************

# Importando a biblioteca OpenCV
import cv2
import numpy as np

# Imagem
aquivo = "./imagens/raposa.jpg"

# Carregando a imagem
imagem = cv2.imread(aquivo)

# Convertendo uma imagem
imagem_gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
cv2.imshow("Imagem base", imagem_gray)
print(imagem_gray.shape)
cv2.waitKey(0) 
