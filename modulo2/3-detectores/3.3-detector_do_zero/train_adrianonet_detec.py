# Estrutura básida para projetos de Machine Learning e Deep Learning
# Por Adriano Santos.

import torch
from torch import nn, optim
from utils.dataset import DataSetDetector
from utils.util import FixedHeightResize
import torchvision.transforms as transforms
from models import AdrianoNet
from torch.autograd import Variable
import torch.nn.functional as F

# Modulos para auxilio na estrutura do projeto.
from tqdm import tqdm
import argparse
import logging
import numpy as np

def main(parser):
    # ************************************ DADOS ***************************************************
    tamanho_da_imagem = 256

    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu"  
    device = torch.device(dev) 

    # Operacoes de preprocessamento e augumentacao
    composicao_de_transformacao = transforms.Compose([
            FixedHeightResize(tamanho_da_imagem),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),])

    # Dataset de treinamento e validacao.
    train = DataSetDetector(True, composicao_de_transformacao, "dataset/completo")
    val = DataSetDetector(False, composicao_de_transformacao, "dataset/completo")

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(dataset=train, shuffle=True, batch_size=parser.batch_size)
    validation_loader = torch.utils.data.DataLoader(dataset=val, batch_size=parser.batch_size)

    # ************************************* REDE ************************************************
    model = AdrianoNet(2, True)
   # model.to(device)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    otimizador = torch.optim.Adam(parameters, lr=0.006)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(otimizador, 200)
    best_acc = 0

    # ************************************ TREINAMENTO E VALIDACAO ********************************************
    for epoch in range(1, parser.epochs):
        logging.info('Treinamento: {}'.format(str(epoch)))
        model.train()
        total = 0
        sum_loss = 0
        for step, (X, bbox, y) in tqdm(enumerate(train_loader), total=len(train_loader)):
            otimizador.zero_grad()

            '''        
            X = X.to(device)
            bbox = bbox.to(device)
            y = y.to(device)
            '''
            yPred, bboxPred = model(X)

            loss_class = F.cross_entropy(yPred, y, reduction="sum")
            loss_bb = F.l1_loss(bboxPred, bbox, reduction="none").sum(1)

            loss_bb = loss_bb.sum()
            loss = loss_class + loss_bb/parser.C
    
            loss.backward()
            otimizador.step()
            lr_scheduler.step()
            total += step
            sum_loss += loss.item()

        train_loss = sum_loss/total

        logging.info('Validacao: {}'.format(str(epoch)))
        model.eval() 
        val_loss = 0
        acertos = 0
        with torch.no_grad():
            for step, (X, bbox, y) in tqdm(enumerate(validation_loader), total=len(validation_loader)):
                '''
                X = X.to(device)
                bbox = bbox.to(device)
                y = y.to(device)
                '''

                yPred, bboxPred = model(X)
                
                loss_class = F.cross_entropy(yPred, y, reduction="sum")
                loss_bb = F.l1_loss(bboxPred, bbox, reduction="none").sum(1)

                loss_bb = loss_bb.sum()
                loss = loss_class + loss_bb/parser.C

                predito = torch.max(yPred,1)[1]
                acertos += (predito == y).sum()
                val_loss += loss.item()
            
            # Imprime métricas
            acc = float(acertos) / (len(validation_loader)*parser.batch_size)
            print("Train loss error: {:.4f}, Val loss error: {:.4f}, Accuracy:{:.4f}".format(train_loss, val_loss, acc))
            
            print(acc, best_acc)
            if acc > best_acc:
                # Imprime mensagem
                print("Um novo modelo foi salvo")
                # Nome do arquivo dos pesos
                pesos = "{}/{}.pt".format(parser.dir_save,str("best"))
                # Salvar os pesos
                chkpt = {'epoch': epoch,'model': model.state_dict()} 
                torch.save(chkpt, pesos)
                best_acc = acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dir_save', default="./pesos")
    parser.add_argument('--dir_root', default="./dataset/completo")
    parser.add_argument('--C', type=int, default=1000)
    parser = parser.parse_args()

    # Main function.
    main(parser)


