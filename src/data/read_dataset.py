# -*- coding: utf-8 -*-
from pathlib import Path
from multiprocessing import Pool  # for reading the CSVs faster

import torch
import torch.utils.data
import numpy as np
import pandas as pd
from scipy.io import wavfile


class MosquitoDataset(torch.utils.data.Dataset):
    """Mosquito dataset for PyTorch"""
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.x)

    def __getitem__(self, index):
        'Generates one sample of data'
        x = self.x[index]
        y = self.y[index]

        return x, y


def my_read_csv(filename):
    # Helper function for the parellel load_csvs
    return pd.read_csv(filename)


def read_all_csvs(folder_path):
    """Reads and joins all our CSV files into one big dataframe.
    We do it in parallel to make it faster, since otherwise it takes some time.
    Idea from: https://stackoverflow.com/questions/36587211/easiest-way-to-read-csv-files-with-multiprocessing-in-pandas

    """
    # Get csv names
    path = Path(folder_path)
    files = [str(x) for x in path.iterdir() if "csv" in x.name]

    # set up your pool
    pool = Pool(processes=1)
    df_list = pool.map(my_read_csv, files)
    pool.close()

    # reduce the list of dataframes to a single dataframe
    return pd.concat(df_list, ignore_index=True)


def my_read_wav(filename):
    return np.hstack((wavfile.read(filename), np.array([filename])))


def read_all_wavs(folder_path, pool_size=None, sampling_rate=44100,
                  testing=False, test_size=1000):
    """Reads and joins all our wavs files into one big dataframe.
    We do it in parallel to make it faster, since otherwise it takes some time.
    Idea from: https://stackoverflow.com/questions/36587211/easiest-way-to-read-csv-files-with-multiprocessing-in-pandas

    RETURNS:
    --------
        all_wavs: A list of tuples, first element being sampling rate,
                    second numpy array containing the data
    """
    # Get csv names
    path = Path(folder_path)
    files = [str(x) for x in path.rglob("*") if "wav" in x.name]

    if testing:
        # Get a random sample of the data
        idx = np.random.randint(len(files), size=test_size)
        files = np.array(files)
        files = list(files[idx])

    # set up your pool
    pool = Pool(pool_size)
    all_wavs = np.array(pool.map(my_read_wav, files))

    # Check all files have the same sampling rate
    return all_wavs
    different_sr = (all_wavs[:, 0] != sampling_rate).sum()
    if different_sr:
        raise ValueError(f"We have {different_sr}" +
                         "files with a different sampling rate!")
    return all_wavs
