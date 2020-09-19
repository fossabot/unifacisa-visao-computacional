from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np

class DataSet(Dataset):
    def __init__(self, imagens, coordenadas, composicao_de_transformacao):
        self.imagens = imagens
        self.coordenadas = coordenadas
        self.composicao_de_transformacao = composicao_de_transformacao

    def __getitem__(self,index):    
        imagem, coordenadas = self.imagens[index], self.coordenadas[index]
        imagem = self.composicao_de_transformacao(imagem)
        return imagem, coordenadas
 
    def __len__(self):
        return len(self.imagens)