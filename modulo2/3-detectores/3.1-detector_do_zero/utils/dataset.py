# Estrutura básica para projetos de Machine Learning e Deep Learning
# Por Adriano Santos.

from torch.utils.data import Dataset
import utils.util as util
import pandas as pd
import os
import numpy as np

class DataSet(Dataset):
    def __init__(self, train=False, transforms=None, raiz="dataset/completo"):

        arquivo = "utils/dados_de_treinamento_classificador.txt" if train else "utils/dados_de_validacao_classificador.txt" 
        dataset = util.obtemDataFrame(arquivo)
        self.raiz = raiz
        self.imagens = dataset.iloc[:,0]
        self.classes = dataset.iloc[:,-1]
        
        self.transforms = transforms

    def __getitem__(self,index):    
        imagem, target = self.imagens[index], self.classes[index]
        imagem = os.path.join(self.raiz, imagem)
        imagem = util.obtemImagem(imagem)
        # Obs: cuidado com a sequencia utilizada na transformacao
        data = self.transforms(imagem)
        return data, target
 
    def __len__(self):
        return len(self.imagens)


class DataSetDetector(Dataset):
    def __init__(self, train=False, transforms=None, raiz="dataset/completo"):

        arquivo_base = "utils/dataset_para_deteccao.csv"
        X_train, X_inter, y_train, y_inter = util.obtemDataSet(arquivo_base, False)

        if train:
            imagens = X_train["imagem"].values
            classes = y_train.values
            boxes = self._bounding_box(X_train)
        else:
            imagens = X_inter["imagem"].values
            classes = y_inter.values
            boxes = self._bounding_box(X_train)
  
        self.raiz = raiz
        self.imagens = imagens
        self.classes = classes
        self.boxes = boxes
        
        self.transforms = transforms

    def __getitem__(self,index):    
        arquivo, box, target = self.imagens[index], self.boxes[index], self.classes[index]
        arquivo = os.path.join(self.raiz, arquivo)

        imagem = util.obtemImagem(arquivo)        
        data = self.transforms(imagem)

        return data, box, target
 
    def __len__(self):
        return len(self.imagens)

    def _bounding_box(self,df):
        novo_bbox = []
        for _, row in df.iterrows():
            novo_bbox.append(np.array([row["x1"], row["y1"], row["x2"], row["y2"]], dtype=np.float32))
        return novo_bbox    
    