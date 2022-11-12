# -*- coding: utf-8 -*-
"""
    Ten plik zawiera kod źródłowy modelu ( sieć LSTM )
"""
__version__ = '0.0.1'
__author__ = 'Raganella'

import pytorch_lightning as pl
import torch
from torch import nn
import torchmetrics

class LSTMModel(pl.LightningModule):
    """ Klasa typu LightningModule

    Parametry:
        point_dim: ilość współrzędnych jednego punktu [int]
        hidden_dim: wielkość jednej ukrytej warstwy [int]
        sequence_len: długość trajektorii cząstki [int]
        label_size: długość jednej etykiety [int]
        num_layers: liczba warstw LSTM [int]
        
    Atrybuty:
        accuracy: metryka mierząca dokładność modelu [torchmetrics]
        num_layers: liczba warstw LSTM [int]
        sequence_len: długość trajektorii cząstki [int]
        hidden_dim: wielkość jednej ukrytej warstwy [int]
        lstm: sekwencja warstw sieci LSTM [LSTM]
        linear: wartswa liniowa [Linear]
    """
    
    def __init__(self, point_dim, hidden_dim, sequence_len, label_size, num_layers):
        ''' Inicjalizacja modelu sieci '''
        super(LSTMModel, self).__init__()
        #Metryki
        self.accuracy = torchmetrics.Accuracy()
        #Architektura
        self.num_layers = num_layers
        self.sequence_len = sequence_len
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(point_dim, hidden_dim, num_layers=self.num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, label_size)
        
    def forward(self, seq, hidden=None):
        ''' Funkcja definiująca obliczenia w trakcie kolejnego wywołania '''
        if hidden is None:
            self.init_hidden(100)
        lstm_out, hidden = self.lstm(seq, hidden)
        pred_labels = self.linear(lstm_out[:,-1])
        return pred_labels, hidden
        
    
    def training_step(self, batch, batch_idx):
        '''Funkcja wywoływana w funkcji fit()'''
        sequences, labels = batch
        pred_labels, _ = self(sequences)
        loss = nn.BCEWithLogitsLoss()
        loss_val = loss(pred_labels, labels.float())
        self.log_dict({'train_loss': loss_val, 'trai_acc': self.accuracy(pred_labels, labels)},
                      on_step=False, on_epoch=True, logger=True)
        return loss_val
    
    def validation_step(self, batch, batch_idx):
        '''Funkcja wywoływana w funkcji fit()'''
        sequences, labels = batch
        pred_labels, _ = self(sequences)
        loss = nn.BCEWithLogitsLoss()
        loss_val = loss(pred_labels, labels.float())
        self.log_dict({'val_loss': loss_val, 'val_acc': self.accuracy(pred_labels, labels)},
                      on_step=False, on_epoch=True, logger=True)
        return loss_val
     
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        '''Funkcja wywoływana w funkcji predict()'''
        sequences, labels = batch
        pred_labels = self(sequences)
        return pred_labels
    
    def configure_optimizers(self):
        ''' Funkcja zwracająca optimalizator oraz współczynnik szybkości uczenia '''
        return torch.optim.Adam(self.parameters(), lr=0.001)
    
    def init_hidden(self, batch_size):
        ''' Funkcja inicjalizująca pierwszą warstwę sieci '''
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        return h_0, c_0