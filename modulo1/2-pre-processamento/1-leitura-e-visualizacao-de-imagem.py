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

# Dimensoes da imagem
print("Altura: {} , Largura: {}, Canais: {}".format(image.shape[0], image.shape[1], image.shape[2]))

# Visualizando a imagem
cv2.imshow("Visao Computacional", image)
cv2.waitKey(0) 
