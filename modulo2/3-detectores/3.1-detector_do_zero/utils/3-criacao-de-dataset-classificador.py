# ****************************************************************
# AULA: Visão Computacional
# Prof: Adriano A. Santos, DSc.
# ****************************************************************
import sys
sys.path.append('../')
import util

import argparse
import logging
from tqdm import tqdm
import sys
import datetime
import os

# Variaveis
arquivo_base = "./dataset_com_classes.csv"

arquivos = ["treinamento", "validacao", "teste"]

# Inicia log
logging.basicConfig(level=logging.DEBUG)

# Obtem informacoes para avaliar o tempo de processamento
inicio = datetime.datetime.now()
logging.info("Iniciando processo em: {}".format(inicio))

# Obtem dados separados
treinamento,validacao,teste = util.obtemDataSetClassificador(arquivo_base)
dados = [treinamento,validacao,teste]

try:
    for i, tipo in enumerate(arquivos):
        arquivo = "./dados_de_{}_classificador.txt".format(tipo) 
        with open (arquivo, "w") as arquivo_de_dados:
            l, c = dados[i].shape
            for j in range(l):
                linha = " ".join(str(item) for item in dados[i][j])
                arquivo_de_dados.write("{}\n".format(linha))
except:
    logging.error("Erro ao processar criacao de arquivo de dados.")
    sys.exit()

# Fim de processo
fim = datetime.datetime.now()
tempo_de_processamento = (fim-inicio).total_seconds()
logging.info("Tempo de processamento: {}".format(str(tempo_de_processamento)))