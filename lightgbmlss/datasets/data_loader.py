from importlib.resources import files

import pandas as pd


def _read_csv(name):
    with files(__package__).joinpath(name).open("rb") as f:
        return pd.read_csv(f)


def load_simulated_gaussian_data():
    """
    Returns train/test dataframe of a simulated example.

    Contains the following columns:
        y              int64: response
        x              int64: x-feature
        X1:X10         int64: random noise features

    """
    train_df = _read_csv("gaussian_train_sim.csv")

    test_df = _read_csv("gaussian_test_sim.csv")

    return train_df, test_df


def load_simulated_studentT_data():
    """
    Returns train/test dataframe of a simulated example.

    Contains the following columns:
        y              int64: response
        x              int64: x-feature
        X1:X10         int64: random noise features

    """
    train_df = _read_csv("studentT_train_sim.csv")

    test_df = _read_csv("studentT_test_sim.csv")

    return train_df, test_df
