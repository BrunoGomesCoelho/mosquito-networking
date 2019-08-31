# -*- coding: utf-8 -*-
import logging

import torch
import numpy as np


def transform_torch(data_vector, device, from_numpy=True):
    # Reshape vectors
    new_vector = []
    for x in data_vector[:2]:
        new_vector.append(x.values.reshape(len(x), 1, -1))
    for x in data_vector[2:]:
        new_vector.append(x.values.reshape(-1, 1))


    if from_numpy:
        logger = logging.getLogger(__name__)
        logger.warning("From numpy option does not send to device!")
        return [torch.from_numpy(x).float() for x in new_vector]
    else:
        return [torch.tensor(x, device=device).float() for x in new_vector]


def get_train_test(df, x_cols=None, division="training", target="label",
                   subtract_mean=True):
    if x_cols is None:
        x_cols = [x for x in range(22049 + 1)]
    train = df.query(f"{division} == 1.0")
    test = df.query(f"{division} != 1.0")

    x_train = train[x_cols]
    y_train = train[target]

    x_test = test[x_cols]
    y_test = test[target]

    if subtract_mean:
        mean = x_train.mean()
        x_train = x_train - mean
        x_test = x_test - mean

    return x_train, x_test, y_train, y_test


def reduce_mem_usage(df):
    """Famous kaggle reduce mem usage script.

    NOT MINE - taken from https://www.kaggle.com/gemartin/load-data-reduce-memory-usage

    Iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df
