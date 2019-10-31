# -*- coding: utf-8 -*-
import gc
import click
import logging
from pathlib import Path
#from dotenv import find_dotenv, load_dotenv

import numpy as np
import pandas as pd
from librosa.util import pad_center

from src.data.read_dataset import read_all_wavs, read_all_csvs
from src.data import util


def main(conversions=["zero"], reduce_mem_usage=False, subsample=0, save=False,
        base_dir="../../data"):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    valid_conversions = ["zero", "repeat", "resample"]
    if any([x not in valid_conversions for x in conversions]):
        raise ValueError("Invalid conversion included! Options are:" + \
                         f"{valid_conversions}")

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # Check for using less data
    if subsample:
        data = read_all_wavs(f"{base_dir}/raw/dadosBruno/", testing=True,
                             test_size=subsample)
    else:
        data = read_all_wavs(f"{base_dir}/raw/dadosBruno/")

    df = read_all_csvs(f"{base_dir}/raw/")
    logger.info('load intitial data')

    # Process the name column
    wav_df = pd.DataFrame(data[:, 2].copy(), columns=["name"])
    process_name(wav_df)

    # Join sizes with original data
    df["label"] = df["label"].astype(int)
    output_df = process_wav_length(data[:, 1], wav_df, df, conversions)
    logger.info('finished processing into 1 df')

    if reduce_mem_usage:
        logger.warning('This seems to take longer that it is worth it!')
        logger.info('Trying to reduce memory usage')
        output_df = util.reduce_mem_usage(output_df)

    # Call the garbage collector manually
    gc.collect()

    if save:
        save_file = f"{base_dir}/processed/data.csv"
        logger.warning('Saving can take a really long time!')
        logger.info(f'Saving to {save_file}')
        output_df.to_csv(save_file, index=False)
        logger.info('finished saving')

    binarize_labels(output_df, logger)

    logger.info("Saving intermediate google colab CSV")
    output_df[["label", "training", "original_name"]].to_csv(f"{base_dir}/interim/file_names.csv",
                                                             index=False)

    return output_df


def binarize_labels(output_df, logger=None, keep_value=1.0):
    """Maps any non 1.0 label to 0.0 so we can consider it a binary problem
        with 2 classes, 0 and 1

    Modifies our output_df in-place
    """
    if logger is not None:
        logger.info('Mapping any non 1.0 label into 0')
    idx = output_df["label"] != keep_value
    output_df.loc[idx, "label"] = 0.0



def process_name(wav_df):
    """Process a given dataset, changing the name of the line to match
        the name of the file so we can join them later.

    Modifies our wav_df in-place
    """
    offset = len("../data/raw/dadosBruno/t04/t04_")
    wav_df["original_name"] = wav_df["name"].copy(deep=True)
    wav_df["file"] = wav_df["name"].str.slice(offset).apply(lambda x: x[::-1])
    wav_df["file"] = wav_df["file"].str.replace("_", "/", n=2).apply(lambda x: x[::-1])


def process_wav_length(wav_data, filenames, df, conversion="zero",
                       seconds=0.25, sr=44100, resample_size=0.125, 
                       testing=False):
    """Process all audios to have the same length.

    conversion:
        zero - We ignore all audios with a bigger size and 0-pad the ones that have less
        repeat - we repeat the audio as many times as necessary to fill the vector
        resample - we resample the audio to a given size
    """
    amount_samples = int(seconds*sr)
    sizes = np.vectorize(len)(wav_data)

    idx = sizes <= amount_samples
    processed_wav_data = wav_data[idx]

    if conversion == "zero":
        new_wavs = np.asarray([pad_center(a, amount_samples)
                              for a in processed_wav_data])
    elif conversion == "repeat":
        new_wavs = np.asarray([np.resize(a, amount_samples)
                               for a in processed_wav_data])
    elif conversion == "rescale":
        raise ValueError("TO-DO")

    new_df = pd.DataFrame(new_wavs)
    new_df["file"] = filenames.loc[idx, "file"].values
    new_df["original_name"] = filenames.loc[idx, "original_name"].values

    full_df = pd.merge(df, new_df, left_on="file", right_on="file",
                       validate="1:1", how="right")

    return full_df


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()
