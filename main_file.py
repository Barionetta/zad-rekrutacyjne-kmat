# -*- coding: utf-8 -*-
"""
    To jest główny plik projektu
"""
__version__ = '0.0.1'
__author__ = 'Raganella'

import numpy as np
import torch
from pytorch_lightning import Trainer, loggers
import pandas as pd
import lstmmodel
import dataset
import visualization

#Ustawianie domyślnego urządzenia
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
torch.multiprocessing.set_start_method('spawn', force=True)

#HIPERPARAMETRY
'''
    batch_size - wielkość "kubełka", które trafia do modelu
    sequence_lenght - długość trajektorii cząstki
    epochs - liczba epok
    label_size - długość jednej etykiety
    num_layers - liczba warstw LSTM
    point_dim - ilość współrzędnych jednego punktu
    hidden_dim - wielkość jednej ukrytej warstwy
'''

batch_size = 100
sequence_lenght = 300
epochs = 400
labels_size = 5
num_layers = 2
point_dim = 2
hidden_dim = 8

def main():
    
    ''' Funkcja główna programu'''
    #Ładowanie danych
    train_data = torch.from_numpy(np.load('data/X_train.npy')).to(device).float()
    train_labels = torch.from_numpy(np.load('data/y_train.npy')).to(device).int()
    val_data = torch.from_numpy(np.load('data/X_val.npy')).to(device).float()
    val_labels = torch.from_numpy(np.load('data/y_val.npy')).to(device).int()
    test_data = torch.from_numpy(np.load('data/X_test.npy')).to(device).float()
    # Sprawdzanie, jaki jest rozkład klas w zbiorze danych
    num_train_labels = torch.argmax(train_labels, dim=1).to(device)
    visualization.draw_class_occurrence_plot(num_train_labels)
    num_val_labels = torch.argmax(val_labels, dim=1).to(device)
    visualization.draw_class_occurrence_plot(num_val_labels)
    # Załadowanie modułu danych oraz modelu
    data_module = dataset.ParticlesDataModule(batch_size,train_data,train_labels
                                              ,val_data,val_labels, test_data)
    logger = loggers.CSVLogger("lstm_logs", name="lstm_metrics")
    model = lstmmodel.LSTMModel(point_dim, hidden_dim, sequence_lenght,
                                labels_size, num_layers).to(device)
    # Jeżeli model nie był trenowany - trenowanie modelu
    trainer = Trainer(logger=logger, max_epochs=epochs, accelerator='gpu', devices=1)
    trainer.fit(model, datamodule=data_module)
    # Jeżeli model był trenowany - załadowanie modelu
    checkpoint = torch.load('model\epoch=399-step=196000.ckpt', map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    
    # WALIDACJA
    model.to(device)
    model.eval()
    with torch.no_grad():
        outputs,_ =model(val_data.to(device))
        _, validations = torch.max(outputs.to(device), 1)
    visualization.confusion_matrix_and_accuracy(validations, num_val_labels, 5, device)
    
    # TEST
    with torch.no_grad():
        outputs,_ = model(test_data.to(device))
        _, predictions = torch.max(outputs.to(device), 1)
    df = pd.DataFrame(predictions.cpu().numpy())
    df.to_csv("submission.csv")
    
    # Prezentacja metryk
    stats = pd.read_csv('model/metrics.csv')
    visualization.plot_acc_and_loss(stats)

if __name__ == "__main__":
    main()
       