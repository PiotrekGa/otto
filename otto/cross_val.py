import polars as pl
from tqdm import tqdm


class CONFIG:
    data_path = 'data/raw/'
    folds = [2, 3]


def main(config):
    scores = []
    for fold in tqdm(config.folds):
        df = pl.read_parquet(f'{config.data_path}valid{fold}__test.parquet')


if __name__ == '__main__':
    main(CONFIG)
