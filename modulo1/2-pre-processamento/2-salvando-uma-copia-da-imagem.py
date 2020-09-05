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

# Salvando a imagem
cv2.imwrite("./imagens/nova_raposa.jpg", image)
cv2.waitKey(0) 
