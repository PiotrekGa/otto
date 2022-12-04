import pandas as pd
import numpy as np


def parquet_to_npz(path, file_name):
    df = pd.read_parquet(f'{path}{file_name}.parquet')
    ts_min = df.groupby('session').min()['ts']
    df['ts_min'] = df.session.map(ts_min)
    df['ts'] = df['ts'] - df['ts_min']
    df = df.loc[:, ['aid', 'ts', 'type']]
    df = df.to_numpy()
    np.savez(f"{path}{file_name}.npz",
             aids=df[:, 0].tolist(), ts=df[:, 1].tolist(), ops=df[:, 2].tolist())


if __name__ == "__main__":
    path = 'data/raw/'
    files = ['train']
    for file in files:
        print(f'processing {file}')
        parquet_to_npz(path, file)
