# -*- coding: utf-8 -*-
"""
    Ten plik zawiera funkcje związane z
    przygotowaniem danych do przetworzenia przez model
"""
__version__ = '0.0.1'
__author__ = 'Raganella'

import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

class ParticlesDataset(Dataset) :
    """ Klasa będąca zbiorem danych

    Atrybuty:
        data: Tensor z danymi
        labels (opcjonalny): Tensor z etykietami 
    """
    
    def __init__(self, data, labels=None):    
        ''' Inicjalizacja zbioru danych '''
        super().__init__()
        self.data = data
        if labels is not None:
            self.labels = labels
        
    def __getitem__(self, index):
        ''' Metoda zwracająca element zbioru o indeksie index '''
        X = self.data[index]
        if self.labels is not None:
            y = self.labels[index]
            return X, y
        return X
    
    def __len__(self):
        ''' Metoda zwracająca długość obiektu '''
        return len(self.data)

class ParticlesDataModule(pl.LightningDataModule):
    """ Klasa będąca modułem danych

    Atrybuty:
        batch_size: wielkość "kubełka", które trafia do modelu
        train_data: tensor z danymi treningowymi
        train_labels: tensor z etykietami do danych treningowych
        val_data: tensor z danymi walidacyjnymi
        val_labels: tensor z etykietami do danych walidacyjnych
        test_data: tensor z danymi testowymi
        train_dataset: zbiór danych (ParticlesDataset) z danymi treningowymi
        val_dataset: zbiór danych (ParticlesDataset) z danymi walidacyjnymi
        test_dataset: zbiór danych (ParticlesDataset) z danymi testowymi
    """
    def __init__(self, batch_size, train_data, train_labels,
                 val_data, val_labels, test_data=None):
        ''' Inicjalizacja modułu danych '''
        super().__init__()
        self.batch_size = batch_size
        self.train_data = train_data
        self.train_labels = train_labels
        self.val_data = val_data
        self.val_labels = val_labels
        self.test_data = test_data
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def setup(self, stage: str):
        ''' Funkcja do tworzenia zbiorów danych '''
        if stage == "fit":
            self.train_dataset = ParticlesDataset(self.train_data, self.train_labels)
            self.val_dataset = ParticlesDataset(self.val_data, self.val_labels)
        if stage == "predict":
            self.test_dataset = ParticlesDataset(self.test_data)
    
    # Funkcje do ładowania zbiorów danych
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=0)

    def predict_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=0)

