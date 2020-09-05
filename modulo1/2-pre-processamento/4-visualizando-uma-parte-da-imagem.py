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

# Obtendo nova imagem e visualizando
nova_raposa = image[54:305, 135:343] #y, x
cv2.imshow("Parte", nova_raposa)
cv2.waitKey(0) 
