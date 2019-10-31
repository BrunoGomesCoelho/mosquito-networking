# -*- coding: utf-8 -*-
from pathlib import Path
from multiprocessing import Pool  # for reading the CSVs faster

import torch
import torch.utils.data
import numpy as np
import pandas as pd
from scipy.io import wavfile


def read_temperature(temperature, conversion):
    from src.data.make_dataset import (process_name, binarize_labels,
                                       process_wav_length)
    """Reads all data of a given temperature.

        Valid options are temperature = t02, ..., t06"""
    folder_path = f"../data/raw/dadosBruno/{temperature}/" 
    data = read_all_wavs(folder_path)
    df = pd.read_csv(f"../data/raw/{temperature}.csv")
    
    # Process the name column
    wav_df = pd.DataFrame(data[:, 2].copy(), columns=["name"])
    process_name(wav_df)

    # Join sizes with original data
    output_df = process_wav_length(data[:, 1], wav_df, df, 
                                   conversion=conversion)
    output_df["label"] = output_df["label"].astype(int)
    binarize_labels(output_df) # binarize the labels
    
    return output_df


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


def read_all_wavs(folder_path, pool_size=1, sampling_rate=44100,
                  testing=False, test_size=1000):
    """Reads and joins all our wavs files into one big dataframe.
    We do it in parallel to make it faster, since otherwise it takes some time.
    Idea from: https://stackoverflow.com/questions/36587211/easiest-way-to-read-csv-files-with-multiprocessing-in-pandas

    PARAMETERS:
    -----------
    pool_size : Amount of parallel processes.


    RETURNS:
    -----------
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
