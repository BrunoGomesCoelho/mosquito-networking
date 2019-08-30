# -*- coding: utf-8 -*-
from pathlib import Path
from multiprocessing import Pool  # for reading the CSVs faster

import numpy as np
import pandas as pd
from scipy.io import wavfile



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
    pool = Pool()
    df_list = pool.map(my_read_csv, files)

    # reduce the list of dataframes to a single dataframe
    return pd.concat(df_list, ignore_index=True)


def my_read_wav(filename):
    return np.hstack((wavfile.read(filename), np.array([filename])))


def read_all_wavs(folder_path, sampling_rate=44100,
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
        print(f"We have {len(files)} wavs")
        files = files[:test_size]

    # set up your pool
    pool = Pool()
    all_wavs = np.array(pool.map(my_read_wav, files))

    # Check all files have the same sampling rate
    different_sr = (all_wavs[:, 0] != sampling_rate).sum()
    if different_sr:
        raise ValueError(f"We have {different_sr}" +
                         "files with a different sampling rate!")
    return all_wavs
