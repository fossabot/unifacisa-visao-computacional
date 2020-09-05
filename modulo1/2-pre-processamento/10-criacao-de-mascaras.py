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
(cX, cY) = (w // 2, h // 2)
aux = 150

# Criacao da mascara  
mascara = np.zeros((h, w), dtype = "uint8")
cv2.rectangle(mascara, (cX - aux, cY - aux), (cX + aux , cY + aux), 255, -1)
cv2.imshow("Mascara", mascara)
cv2.waitKey(0) 
