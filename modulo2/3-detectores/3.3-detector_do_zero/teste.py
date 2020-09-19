import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps
import matplotlib.pyplot as plt 
import utils.util as util

from utils.daod import DAOD

dd = DAOD()

arquivo = "./utils/raposa.jpg"

imagem_base = cv2.imread(arquivo)

lines = util.obtemLinhas(arquivo.replace(".jpg", ".txt"))

for annt in lines:
    box = annt.split()
    box = box[1:]

    imagem_box = dd.create_mask(imagem_base, box)



imagem, imagem_box = dd.random_crop_image_bbox(imagem_base.copy(), imagem_box.copy(), 1)
imagemB, imagemB_box = dd.resize(imagem, 200), dd.resize(imagem_box, 200)


box = dd.convert_mask_to_bbox(imagem_box)
imagem = dd.draw_box(imagem, box)

boxb = dd.convert_mask_to_bbox(imagemB_box)
imagemB = dd.draw_box(imagemB, boxb)

cv2.imshow("Imagem base ", imagem)
cv2.imshow("Imagem resize ", imagemB)
cv2.waitKey(0) 
