# ****************************************************************
# AULA: Visão Computacional
# Prof: Adriano A. Santos, DSc.
# ****************************************************************
#import sys
#sys.path.append('../')
import util

import argparse
import logging
from tqdm import tqdm
import sys
import datetime
import os
import numpy as np

# Variaveis
diretorio_base = "../dataset/completo/"
dataset = "./dataset_para_deteccao.csv"
tipo = "png"

# Inicia log
#logging.basicConfig(level=logging.DEBUG)

# Obtem informacoes para avaliar o tempo de processamento
inicio = datetime.datetime.now()
logging.info("Iniciando processo em: {}".format(inicio))
try:
    arquivos = util.obtemTodosOsArquivos(diretorio_base, tipo, False)

    with open (dataset, "w") as arquivo_de_dados:
        arquivo_de_dados.write("{};{};{};{};{};{}\n".format("imagem","x1","y1","x2","y2","classe"))
        for _, arquivo in tqdm(enumerate(arquivos), total=len(arquivos)): # Cria barra de progressao

            nome_arquivo = util.obtemNomeDoArquivo(arquivo)

            imagem = util.obtemImagem(arquivo)

            anotacoes = util.obtemAnotacao(arquivo)

            classe = anotacoes[0]

            box = anotacoes[1:]

            imagem = np.asarray(imagem)

            box = util.converte_xywh_para_xyxy(imagem, box)

           # Formacao ( caminho, w, h, classe)
            arquivo_de_dados.write("{};{};{};{};{};{}\n".format(nome_arquivo,box[0],box[1],box[2],box[3],classe))
except:
    logging.error("Erro ao processar criacao de arquivo de dados.")
    sys.exit()

# Fim de processo
fim = datetime.datetime.now()
tempo_de_processamento = (fim-inicio).total_seconds()
logging.info("Tempo de processamento: {}".format(str(tempo_de_processamento)))