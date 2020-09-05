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
image = cv2.imread(aquivo)

# Obtem valores para calcular o valor do ponto central da imagem
(h, w) = image.shape[:2]

# Deslocamento da imagem
matriz = np.float32([[1, 0, 25], [0, 1, 0]])
print(matriz)

nova_imagem = cv2.warpAffine(image, matriz, (w, h))
cv2.imshow("Imagem deslocada", nova_imagem)
cv2.waitKey(0) 
