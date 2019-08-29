# -*- coding: utf-8 -*-
from pathlib import Path
from multiprocessing import Pool  # for reading the CSVs faster

import pandas as pd


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
