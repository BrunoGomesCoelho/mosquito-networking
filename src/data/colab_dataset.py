# -*- coding: utf-8 -*-
import numpy as np
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


class MosquitoDataTemperature(torch.utils.data.Dataset):
    """Mosquito dataset for PyTorch"""
    def __init__(self, x, y, device, scaler, roll=0, sample=22050):
        self.x = x
        self.y = y
        self.device = device
        self.scaler = scaler
        self.roll = roll
        self.sample = sample

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.x)

    def roll_x(self, x):
        """Randomly roll a vector to the left or to the right.
        We fill the otherside with 0s"""
        x = x.reshape(-1)
        roll_idx = int(self.roll*self.sample)
        roll_offset = np.random.randint(-roll_idx, +roll_idx)
        left_extra = 0 if roll_offset <= 0 else roll_offset
        right_extra = 0 if roll_offset >= 0 else -roll_offset
        extra = ((left_extra), (right_extra))

        if roll_offset > 0:
            return np.pad(x, extra, mode="constant")[:-roll_offset]
        elif roll_offset < 0:
            return np.pad(x, extra, mode="constant")[-roll_offset:]
        return x

    def __getitem__(self, index):
        'Generates one sample of data'
        x = self.x[index]
        x = self.scaler.transform(x.reshape(1, -1))
        if self.roll:
            x = self.roll_x(x)
        y = self.y[index]
        return x.reshape(1, -1), y
