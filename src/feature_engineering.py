import pandas as pd


def get_features_and_target(df):
    df = df.copy()

    target = "G3"

    X = df.drop(columns=[target])
    y = df[target]

    return X, y