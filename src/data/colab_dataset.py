# -*- coding: utf-8 -*-
import torch
import torch.utils.data
from librosa.util import pad_center


class MosquitoDatasetColab(torch.utils.data.Dataset):
    """Mosquito dataset for PyTorch"""
    def __init__(self, x, y, device, scaler, samples=22050):
        self.x = x
        self.y = y
        self.device = device
        self.samples = samples
        self.scaler = scaler

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.x)

    def __getitem__(self, index):
        'Generates one sample of data'
        x = self.x[index]
        x = pad_center(x, self.samples)
        x = self.scaler.transform(x.reshape(1, -1))
        y = self.y[index]

        return x.reshape(1, -1), y
