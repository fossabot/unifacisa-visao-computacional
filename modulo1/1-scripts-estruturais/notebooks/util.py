import glob2
import os
from sklearn.model_selection import train_test_split 
import pandas as pd
import cv2
import numpy as np

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

def obtemDataSet(dados, tamanho=0.30):
    df = pd.read_csv(dados, delimiter=";")
    X = df.iloc[:,:-1].values
    y = df.iloc[:,-1]
    # Obtem dados para treinamento e dados intermediarios
    X_train, X_inter, y_train, y_inter = train_test_split(X, y, test_size=tamanho)
    
    # Obtem dados para validacao e teste
    X_val, X_test, y_val, y_test = train_test_split(X_inter, y_inter, test_size=0.5, random_state=1)
    # Agrupa os dados
    treinamento = np.insert(X_train, int(X_train.shape[1]), y_train, axis=1) 
    validacao = np.insert(X_val,int(X_val.shape[1]), y_val,axis=1)
    teste = np.insert(X_test, int(X_test.shape[1]), y_test, axis=1)
    # Treinamento, Validacao e teste
    return treinamento,validacao,teste     
   