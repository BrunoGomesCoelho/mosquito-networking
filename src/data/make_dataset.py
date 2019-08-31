# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import numpy as np
import pandas as pd
from librosa.util import pad_center

from src.data.read_dataset import read_all_wavs, read_all_csvs


"""
@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
"""


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    data = read_all_wavs("../../data/raw/dadosBruno/")
    df = read_all_csvs("../../data/raw/")
    logger.info('load intitial data')

    # Join sizes with original data
    wav_df = pd.DataFrame(data[:, 2].copy(), columns=["name"])

    offset = len("../../data/raw/dadosBruno/t04/t04_")
    wav_df["file"] = wav_df["name"].str.slice(offset).apply(lambda x: x[::-1])
    wav_df["file"] = wav_df["file"].str.replace("_", "/", n=2).apply(lambda x: x[::-1])

    output_df = process_wav_length(data[:, 1], wav_df, df)
    logger.info('finished processing into 1 df')
    return output_df
    #output_df.to_csv("../../data/processed/data.csv", index=False)
    #logger.info('finished saving')


def process_wav_length(wav_data, filenames, df, seconds=0.5, sr=44100):
    """Process all audios to have the same length.

    We ignore all audios with a bigger size and 0-pad the ones that have less
    """
    amount_samples = int(seconds*sr)
    sizes = np.vectorize(len)(wav_data)

    idx = sizes <= amount_samples
    processed_wav_data = wav_data[idx]
    padded_wavs = np.asarray([pad_center(a, amount_samples)
                              for a in processed_wav_data])

    new_df = pd.DataFrame(padded_wavs)
    new_df["file"] = filenames.loc[idx, "file"].values

    full_df = pd.merge(df, new_df, left_on="file", right_on="file",
                       validate="1:1", how="right")

    return full_df


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
