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

# Obtendo pixel
(b, g, r) = image[28, 28]

# Imprimindo valores
print("R: {}, G: {}, R: {}".format(r, g, b))

# Valor esperado: R: 69, G: 84, R: 61
cv2.waitKey(0) 
