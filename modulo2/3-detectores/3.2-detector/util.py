import glob2
import os
from sklearn.model_selection import train_test_split 
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps
import imutils

def obtemTodosOsArquivos(diretorio_base, tipo_de_arquivo, eRecursivo=False):
    regra = "/**/*.{}" if eRecursivo else "/*.{}"
    caminho = diretorio_base + regra.format(tipo_de_arquivo)
    arquivos = glob2.glob(caminho , recursive=eRecursivo)
    return arquivos

def obtemTodosOsDiretorio(diretorio_base, eRecursivo=False):
    return [d for d in os.listdir(diretorio_base) if os.path.isdir(os.path.join(diretorio_base, d))]

def obtemLinhas(arquivo):
    with open(arquivo, "r") as f:
        return [l.strip() for l in f]

def obterLarguraAltura(imagem):
    img = cv2.imread(imagem)
    return img.shape[0], img.shape[1]

def obtemDataSet(arquivo, tamanho=0.30):
    df = obtemDataFrame(arquivo)

    X = df.drop('classe', axis=1)
    y = df.classe

    # Obtem dados para treinamento e dados intermediarios
    X_train, X_inter, y_train, y_inter = train_test_split(X, y, test_size=tamanho)
    
    # Treinamento, Validacao e teste
    return X_train, X_inter, y_train, y_inter     

def obtemNomeDoArquivo(arquivo):
    return Path(arquivo).name

def obtemDataFrame(arquivo):
    return pd.read_csv(arquivo, delimiter=";")

# Para a utilizacao do data Transform
def obtemImagem(imagem):
    return Image.open(imagem).convert('RGB')

def obtemAnotacao(arquivo):
    arquivo = arquivo.replace(".png", ".txt")
    linhas = obtemLinhas(arquivo)
    anotacoes = [item for item in linhas]
    
    box = ""
    for anotacao in anotacoes:
        box = anotacao.split()
    return box

def obtem_coor_x_y_w_h(image, bbox):
    (H, W) = image.shape[:2]
    
    # Get x and y base
    w = float(bbox[2])
    h = float(bbox[3])
    x = float(bbox[0]) 
    y = float(bbox[1])

    # Get x and y base
    x = x - w / 2
    y = y - h / 2

    # Update values according to image dimensions
    x = int(x * W)
    w = int(w * W)
    y = int(y * H)
    h = int(h * H)

    return [x,y,w,h]

# Convert bounding box format from [x, y, w, h] to [y,x,y2,x2]
def converte_xywh_para_xyxy(image, bbox):
    x, y, w, h =  obtem_coor_x_y_w_h(image, bbox)

    x2 = x + w
    y2 = y + h

    return [x,y,x2,y2]


def sliding_window(image, stride=4, size=(50, 50)):
	# Percorre toda a image de acordo com os parametros de stride e o tamanho da janela
	for y in range(0, image.shape[0] - size[1], stride):
		for x in range(0, image.shape[1] - size[0], stride):
			# Retorna as coordenadas x, y e o array que representa a imagem
			yield (x, y, image[y:y + size[1], x:x + size[0]])


def image_pyramid(imagem, escala=2., tamanho=(50, 50)):
	# Retorna a imagem original
	yield imagem
	# Obtem imagens até que o criterio de parada seja obedecido
	while True:
		# compute the dimensions of the next image in the pyramid
		w = int(imagem.shape[1] / escala)
		imagem = imutils.resize(imagem, width=w)

		if imagem.shape[0] < tamanho[1] or imagem.shape[1] < tamanho[0]:
			break
		# retorna uma imagem
		yield imagem


# Funcao baseada em: https://github.com/jrosebr1/imutils/blob/master/imutils/object_detection.py
def non_max_suppression(boxes, probs=None, overlapThresh=0.3):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# if the bounding boxes are integers, convert them to floats -- this
	# is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes
	pick = []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]

	# compute the area of the bounding boxes and grab the indexes to sort
	# (in the case that no probabilities are provided, simply sort on the
	# bottom-left y-coordinate)
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = y2

	# if probabilities are provided, sort on them instead
	if probs is not None:
		idxs = probs

	# sort the indexes
	idxs = np.argsort(idxs)

	# keep looping while some indexes still remain in the indexes list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the index value
		# to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the largest (x, y) coordinates for the start of the bounding
		# box and the smallest (x, y) coordinates for the end of the bounding
		# box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]

		# delete all indexes from the index list that have overlap greater
		# than the provided overlap threshold
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))

	# return only the bounding boxes that were picked
	return boxes[pick].astype("int")
