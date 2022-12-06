import polars as pl
from pathlib import Path


def generate_candidates(fold, config):

    clicked_in_session = RecentEvents(
        fold, 0, 'clicked_in_session', config.data_path)
    clicked_in_session = clicked_in_session.load_candidates_file()

    carted_in_session = RecentEvents(
        fold, 1, 'carted_in_session', config.data_path)
    carted_in_session = carted_in_session.load_candidates_file()

    ordered_in_session = RecentEvents(
        fold, 2, 'ordered_in_session', config.data_path)
    ordered_in_session = ordered_in_session.load_candidates_file()

    candidates = pl.concat(
        [clicked_in_session, carted_in_session, ordered_in_session]).collect()

    candidates = candidates.pivot(
        values='rank', index=['session', 'aid'], columns='name').fill_null(999)
    return candidates


class CandiadateGen():
    def __init__(self, fold, name, data_path):
        self.fold = fold
        self.name = name
        self.data_path = data_path

    def prepare_candidates(self):
        raise NotImplementedError

    def load_candidates_file(self):
        candidate_file = Path(
            f'{self.data_path}candidates/{self.fold}{self.name}.parquet')
        if not candidate_file.is_file():
            self.prepare_candidates()
        return pl.read_parquet(candidate_file.as_posix()).lazy()


class RecentEvents(CandiadateGen):
    def __init__(self, fold, event_type, name, data_path):
        super().__init__(fold=fold, name=name, data_path=data_path)
        self.fold = fold
        self.event_type = event_type

    def prepare_candidates(self):
        df = pl.read_parquet(
            f'{self.data_path}raw/{self.fold}test.parquet').lazy()
        df = df.filter(pl.col('type') == self.event_type)
        df = df.sort(by='ts', reverse=True)
        df = df.select(pl.col(['session', 'aid']))
        df = df.unique(keep='first')
        df = df.select([pl.col(['session', 'aid']), pl.col(
            'aid').cumcount().over("session").alias('rank')])
        df = df.with_column(pl.lit(self.name).alias('name')).collect()
        df.write_parquet(
            f'{self.data_path}candidates/{self.fold}{self.name}.parquet')
