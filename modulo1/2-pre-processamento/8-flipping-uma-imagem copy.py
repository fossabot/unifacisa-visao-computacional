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

# Flip horizontal
imagem_com_flip = cv2.flip(imagem, 1)
cv2.imshow("Horizontal", imagem_com_flip)
cv2.waitKey(0)

# Flip vertical
imagem_com_flip = cv2.flip(imagem, 0)
cv2.imshow("Vertical", imagem_com_flip)
cv2.waitKey(0)

# Flip em ambas as direcoes
imagem_com_flip = cv2.flip(imagem, -1)
cv2.imshow("Ambos", imagem_com_flip)
cv2.waitKey(0)