# -*- coding: utf-8 -*-
"""
    Ten plik zawiera funkcje pomocnicze do wizualizacji danych
"""
__version__ = '0.0.1'
__author__ = 'Raganella'

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torchmetrics import ConfusionMatrix

#Ile z danej klasy
def draw_class_occurrence_plot(labels):
    """Funkcja rysująca wykres słupkowy częstotliwości występowania danej klasy
    
    Argumenty:
        labels: Tensor z etykietami"""
        
    occurrence = torch.unique(labels, return_counts=True)
    sns.set_theme()
    plt.figure(figsize=(13, 7))
    sns.barplot(x=occurrence[0].cpu().numpy(),y=occurrence[1].cpu().numpy(),
                palette='Purples').set(xlabel="Klasa",
                                       ylabel="Częstotliwość występowania",
                                       title="Wykres częstotliwości wystąpienia danej klasy")
    plt.tight_layout()
    plt.show()

def confusion_matrix_and_accuracy(pred_labels, labels, num_labels, device):
    """Funkcja wypisująca dokładność dla każdej klasy oraz rysująca macierz błędów.
    
    Argumenty:
        pred_labels: Tensor z przewidzianymi etykietami
        labels: Tensor z etykietami
        num_labels: liczba klas [int]
        device: urządzenie na którym znajdują się tensory [torch.device]"""
    
    # Obliczenia dla macierzy
    conf_metric = ConfusionMatrix(num_classes=num_labels, normalize='true').to(device)
    conf_matrix = conf_metric(pred_labels, labels).cpu().numpy()
    per_label = conf_matrix.diagonal()
    for i in range(len(per_label)):
        print('Dokładnosć dla klasy {} wynosi {}%'.format(i, per_label[i]*100))
    conf_df = pd.DataFrame(conf_matrix, index = ['attm','crtw','fbm','lw','sbm'],
                           columns = ['attm','crtw','fbm','lw','sbm'])
    # Rysowanie macierzy błędów
    plt.figure(figsize = (13, 7))
    sns.set_theme()
    sns.heatmap(data=conf_df, annot=True,fmt='f', cmap='Purples').set_title("Macierz błędów dla wszystkich klas")
    plt.tight_layout()
    plt.show()

#Rysowanie dokladnosci i funkcji straty
def plot_acc_and_loss(stats_df):
    """Funkcja rysująca wykresy dokładności i straty dla treningu i walidacji.
    
    Argumenty:
        stats_df: obiekt typu DataFrame ze statystykami wczytanymi z pliku csv"""
    
    # Sformatowanie danych
    for_train = pd.DataFrame(stats_df, columns = ['epoch', 'train_loss', 'trai_acc']).dropna()
    for_val = pd.DataFrame(stats_df, columns = ['epoch', 'val_loss', 'val_acc']).dropna()
    # Rysowanie wykresów
    sns.set_theme()
    sns.set_palette('plasma')
    plt.figure(figsize = (13, 7))
    plt.plot(for_train['epoch'], for_train['train_loss'], label = "Trening")
    plt.plot(for_val['epoch'], for_val['val_loss'], label = "Walidacja")
    plt.title("Strata dla treningu i walidacji")
    plt.xlabel("Epoka")
    plt.ylabel("Strata")
    plt.legend()
    plt.tight_layout()
    plt.figure(figsize = (13, 7))
    plt.plot(for_train['epoch'], for_train['trai_acc'], label = "Trening")
    plt.plot(for_val['epoch'], for_val['val_acc'], label = "Walidacja")
    plt.title("Dokładność dla treningu i walidacji")
    plt.xlabel("Epoka")
    plt.ylabel("Dokładność")
    plt.legend()
    plt.tight_layout()
