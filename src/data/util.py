# -*- coding: utf-8 -*-


def get_train_test(df, x_cols=None, target="label", subtract_mean=True):
    if x_cols is None:
        x_cols = [x for x in range(22049 + 1)]
    train = df.query("label == 1.0")
    test = df.query("label != 1.0")

    x_train = train[x_cols]
    y_train = train[target]

    x_test = test[x_cols]
    y_test = test[target]

    if subtract_mean:
        mean = x_train.mean()
        x_train = x_train - mean
        x_test = x_test - mean

    return x_train, x_test, y_train, y_test
