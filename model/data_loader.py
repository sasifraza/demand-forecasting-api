import pandas as pd
import numpy as np

def load_data(path="data/sales_train_validation.csv"):
    df = pd.read_csv(path)
    return df


def prepare_single_series(df):
    # take one product (first row)
    row = df.iloc[0]

    # select only demand columns (d_1 ... d_n)
    demand = row.filter(like="d_")

    # convert to DataFrame
    ts = demand.to_frame(name="sales")
    ts.reset_index(inplace=True)

    # create time index
    ts["day"] = ts["index"].str.replace("d_", "").astype(int)
    ts = ts.sort_values("day")

    # create date range
    ts["date"] = pd.date_range(start="2011-01-29", periods=len(ts))

    ts = ts[["date", "sales"]]

    return ts


def create_features(df):
    df = df.copy()

    # lag features
    df["lag_1"] = df["sales"].shift(1)
    df["lag_7"] = df["sales"].shift(7)
    df["lag_14"] = df["sales"].shift(14)

    # rolling features
    df["rolling_mean_7"] = df["sales"].rolling(7).mean()
    df["rolling_mean_14"] = df["sales"].rolling(14).mean()

    # calendar features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    df = df.dropna()

    return df