# -*- coding: utf-8 -*-
import torch
import torch.utils.data
from scipy.io import wavfile
from librosa.util import pad_center


class MosquitoDatasetColab(torch.utils.data.Dataset):
    """Mosquito dataset for PyTorch"""
    def __init__(self, x, y, device, samples=22050):
        self.x = x
        self.y = y
        self.device = device
        self.samples = samples

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.x)

    def __getitem__(self, index):
        'Generates one sample of data'
        x = wavfile.read(self.x[index])[1]
        x = pad_center(x, self.samples)
        y = self.y[index]

        return x.reshape(1, -1), y
