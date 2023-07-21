import datetime
import os
import pickle

import pandas as pd
import pandas_datareader.stooq as stooq
import yaml


def stockvalues_tokyo(stockcode, start, end, use_ratio=False):
    """
    stockcode: market code of each target stock defined by the Tokyo stock market.
    start, end: datetime object
    """
    # Get index data from https://stooq.com/
    df = stooq.StooqDailyReader(f"{stockcode}.jp", start, end).read()
    df = df.sort_values(by="Date", ascending=True)

    if use_ratio:
        df = df.apply(lambda x: (x - x[0]) / x[0])
    return df


def paneldata_tokyo(stockcodes, start, end, use_ratio=False):
    # Use "Close" value only
    dfs = []
    for sc in stockcodes:
        df = stockvalues_tokyo(sc, start, end, use_ratio)[["Close"]]
        df = df.rename(columns={"Close": sc})
        dfs.append(df)
    df_concat = pd.concat(dfs, axis=1)
    return df_concat


def main():
    with open("./src/setting.yml", "r") as file:
        const = yaml.safe_load(file)

    start = datetime.datetime.strptime(const["start"], "%Y-%m-%d")
    end = datetime.datetime.strptime(const["end"], "%Y-%m-%d")
    stockcodes = const["stock"].keys()
    shared_dirname = "./shared"

    df = paneldata_tokyo(stockcodes, start, end, use_ratio=True)

    if not os.path.exists(shared_dirname):
        os.makedirs(shared_dirname)

    filename = os.path.join(shared_dirname, "stock_data.pkl")
    with open(filename, "wb") as file:
        pickle.dump(df, file)


if __name__ == "__main__":
    main()
