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

# Variaveis
arquivo_base = "./dataset_para_deteccao.csv"

arquivos = ["treinamento", "validacao"]

# Inicia log
logging.basicConfig(level=logging.DEBUG)

# Obtem informacoes para avaliar o tempo de processamento
inicio = datetime.datetime.now()
logging.info("Iniciando processo em: {}".format(inicio))

# Obtem dados separados
X_train, X_inter, y_train, y_inter = util.obtemDataSetDetector(arquivo_base)

# Fim de processo
fim = datetime.datetime.now()
tempo_de_processamento = (fim-inicio).total_seconds()
logging.info("Tempo de processamento: {}".format(str(tempo_de_processamento)))
