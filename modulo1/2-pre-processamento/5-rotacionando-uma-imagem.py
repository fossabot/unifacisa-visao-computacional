# ****************************************************************
# AULA: Visão Computacional
# Prof: Adriano A. Santos, DSc.
# ****************************************************************

# Importando a biblioteca OpenCV
import cv2

# Imagem
aquivo = "./imagens/raposa.jpg"

# Carregando a imagem
image = cv2.imread(aquivo)

# Obtem valores para calcular o valor do ponto central da imagem
(h, w) = image.shape[:2]
centro = (w // 2, h // 2)

# Rotacionando a imagem em 90 graus
matriz = cv2.getRotationMatrix2D(centro, 90, 1.0)
nova_imagem = cv2.warpAffine(image, matriz, (w, h))
cv2.imshow("Imagem rotacionada", nova_imagem)
cv2.waitKey(0) 
