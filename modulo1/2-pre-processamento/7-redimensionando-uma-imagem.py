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
cv2.imshow("Imagem base", imagem)
cv2.waitKey(0) 

# Obtem valores para calcular o valor do ponto central da imagem
(h, w) = imagem.shape[:2]
dim = 400

# Redimensionamento com base na largura
r = dim / w
dim_final = (dim, int(h * r))
imagem_largura = cv2.resize(imagem, dim_final, interpolation = cv2.INTER_AREA)
cv2.imshow("Red. por largura", imagem_largura)
cv2.waitKey(0) 

# Redimensionamento com base na altura
r = dim / h
dim_final = (dim, int(w * r))
imagem_altura = cv2.resize(imagem, dim_final, interpolation = cv2.INTER_AREA)
cv2.imshow("Red. por altura", imagem_altura)
cv2.waitKey(0) 
