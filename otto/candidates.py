import polars as pl
import numpy as np
import scipy
from pathlib import Path
from tqdm import tqdm
from gensim.models import Word2Vec
from gensim.similarities.annoy import AnnoyIndexer
from implicit.nearest_neighbours import bm25_weight
from implicit.bpr import BayesianPersonalizedRanking
from implicit.als import AlternatingLeastSquares


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
        [clicked_in_session, carted_in_session, ordered_in_session])
    candidates = candidates.pivot(
        values='rank', index=['session', 'aid'], columns='name')

    covisit1_clicks = CandsFromSubmission(
        fold=fold, event_type_str='clicks', name='covisit1_clicks', data_path=config.data_path, base_file_name='covisit1', reverse=False)
    covisit1_clicks = covisit1_clicks.load_candidates_file()

    candidates = candidates.join(
        covisit1_clicks, on=['session', 'aid'], how='outer')
    del covisit1_clicks

    covisit1_carts = CandsFromSubmission(
        fold=fold, event_type_str='carts', name='covisit1_carts', data_path=config.data_path, base_file_name='covisit1', reverse=False)
    covisit1_carts = covisit1_carts.load_candidates_file()

    candidates = candidates.join(
        covisit1_carts, on=['session', 'aid'], how='outer')
    del covisit1_carts

    mf1_clicks = CandsFromSubmission(
        fold=fold, event_type_str='clicks', name='mf1_clicks', data_path=config.data_path, base_file_name='matrix_factorization1', reverse=False)
    mf1_clicks = mf1_clicks.load_candidates_file()

    candidates = candidates.join(
        mf1_clicks, on=['session', 'aid'], how='outer')
    del mf1_clicks

    mf1_carts = CandsFromSubmission(
        fold=fold, event_type_str='carts', name='mf1_carts', data_path=config.data_path, base_file_name='matrix_factorization1', reverse=False)
    mf1_carts = mf1_carts.load_candidates_file()

    candidates = candidates.join(
        mf1_carts, on=['session', 'aid'], how='outer')
    del mf1_carts

    mf1_orders = CandsFromSubmission(
        fold=fold, event_type_str='orders', name='mf1_orders', data_path=config.data_path, base_file_name='matrix_factorization1', reverse=False)
    mf1_orders = mf1_orders.load_candidates_file()

    candidates = candidates.join(
        mf1_orders, on=['session', 'aid'], how='outer')
    del mf1_orders

    w2v_window09 = W2VReco(
        fold, 'w2v_window09', config.data_path, '09', 30)
    w2v_window09 = w2v_window09.load_candidates_file(max_rank=5)

    candidates = candidates.join(
        w2v_window09, on=['session', 'aid'], how='outer')
    del w2v_window09

    w2v_window01 = W2VReco(
        fold, 'w2v_window01', config.data_path, '01', 30)
    w2v_window01 = w2v_window01.load_candidates_file(max_rank=5)

    candidates = candidates.join(
        w2v_window01, on=['session', 'aid'], how='outer')
    del w2v_window01

    w2v_window35 = W2VReco(
        fold, 'w2v_window35', config.data_path, '35', 30)
    w2v_window35 = w2v_window35.load_candidates_file(max_rank=5)

    candidates = candidates.join(
        w2v_window35, on=['session', 'aid'], how='outer')
    del w2v_window35

    covisit2 = Covisit(fold=fold, name='civisit2', data_path='../data/', max_cands=30, type_weight={0: 1, 1: 6, 2: 3},
                       days_back=14, before_time=0, after_time=24 * 60 * 60, )
    covisit2 = covisit2.load_candidates_file(max_rank=20)

    candidates = candidates.join(
        covisit2, on=['session', 'aid'], how='outer')
    del covisit2

    covisit3 = CovisitMaster(fold=fold, name='civisit3', data_path='../data/', max_cands=30, type_weight={0: 1, 1: 6, 2: 3},
                             days_back=14, before_time=0, after_time=24 * 60 * 60, left_types=[2], right_types=[2])
    covisit3 = covisit3.load_candidates_file(max_rank=20)

    candidates = candidates.join(
        covisit3, on=['session', 'aid'], how='outer')
    del covisit3

    candidates = candidates.fill_null(999)

    cands_cols = candidates.columns
    cands_cols.remove('aid')
    cands_cols.remove('session')

    candidates = candidates.with_columns(pl.col(cands_cols).cast(pl.UInt16))

    return candidates


class CandiadateGen():

    def __init__(self, fold, name, data_path):
        self.fold = fold
        self.name = name
        self.data_path = data_path

    def prepare_candidates(self):
        raise NotImplementedError

    @ staticmethod
    def filter(df, max_rank):
        columns = df.columns
        columns.remove('session')
        columns.remove('aid')
        df = df.filter(df.select(columns).min(1) <= max_rank)
        return df

    def load_candidates_file(self, max_rank=None):
        candidate_file = Path(
            f'{self.data_path}candidates/{self.fold}{self.name}.parquet')
        if not candidate_file.is_file():
            print(f'preparing candidates {self.fold}{self.name}')
            self.prepare_candidates()
        df = pl.read_parquet(candidate_file.as_posix())
        if max_rank:
            return self.filter(df, max_rank)
        else:
            return df


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


class CandsFromSubmission(CandiadateGen):

    def __init__(self, fold, event_type_str, name, data_path, base_file_name, reverse=False):
        super().__init__(fold=fold, name=name, data_path=data_path)
        self.fold = fold
        self.event_type_str = event_type_str
        self.base_file_name = base_file_name
        self.reverse = reverse

    def prepare_candidates(self):
        df = pl.read_csv(
            f'{self.data_path}raw/{self.fold}{self.base_file_name}.csv').lazy()
        df = df.with_column(pl.col('labels').apply(
            lambda x: [int(i) for i in x.split()]).alias('candidates'))
        df = df.with_column(pl.col('session_type').str.split(
            by='_').alias('session_type2'))
        df = df.with_column(pl.col('session_type2').apply(
            lambda x: int(x[0])).alias('session'))
        df = df.with_column(pl.col('session_type2').apply(
            lambda x: x[1]).alias('type_str'))
        df = df.filter(pl.col('type_str') == self.event_type_str)
        df = df.drop(['session_type', 'labels', 'session_type2'])
        df = df.explode('candidates')
        df = df.with_column(pl.lit(1).alias('one'))
        df = df.with_column(
            (pl.col('one').cumsum(reverse=self.reverse) - 1).over('session').alias(self.name))
        df = df.drop('one')
        df = df.select(
            [pl.col('session').cast(pl.Int32), pl.col('candidates').cast(pl.Int32).alias('aid'), pl.col(self.name).cast(pl.Int32)]).collect()
        df.write_parquet(
            f'{self.data_path}candidates/{self.fold}{self.name}.parquet')


class W2VReco(CandiadateGen):
    def __init__(self, fold, name, data_path, window_str, max_cands):
        super().__init__(fold=fold, name=name, data_path=data_path)
        self.fold = fold
        self.window_str = window_str
        self.max_cands = max_cands

    def prepare_candidates(self):
        df = pl.read_parquet(
            f'{self.data_path}raw/{self.fold}test.parquet').lazy()
        model = Word2Vec.load(
            f'{self.data_path}raw/word2vec_w{self.window_str}.model')

        df = df.unique(subset=['session'], keep='last')
        df = df.with_column((pl.col('aid').cast(
            str) + pl.lit('_') + pl.col('type').cast(str)).alias('aid_str'))
        df = df.with_column(pl.col('type').cast(str)).collect()
        vocab = list(set(model.wv.index_to_key))
        vocab = pl.DataFrame(vocab, columns=['aid_str'])
        df = df.join(vocab, on='aid_str')
        df = df.select(pl.col(['session', 'aid_str']))
        annoy_index = AnnoyIndexer(model, 100)
        cands = []
        for aid_str in tqdm(df.select(pl.col('aid_str').unique()).to_dict()['aid_str']):
            cands.append(self.get_w2v_reco(aid_str, model, annoy_index))
        cands = pl.concat(cands)
        df = df.join(cands, on='aid_str').drop('aid_str')
        df = df.select(pl.col('*').cast(pl.Int32))
        df.write_parquet(
            f'{self.data_path}candidates/{self.fold}{self.name}.parquet')

    def get_w2v_reco(self, aid_str, model, indexer):
        cands = []
        rank_clicks = 0
        rank_carts = 0
        rank_orders = 0

        recos = model.wv.most_similar(aid_str, topn=200, indexer=indexer)
        for reco in recos:
            if len(reco[0]) > 1:
                if reco[0][-1] == '0' and rank_clicks < self.max_cands:
                    cands.append([aid_str, int(reco[0][:-2]),
                                 f'w2v_{self.window_str}_clicks', rank_clicks])
                    rank_clicks += 1
                elif reco[0][-1] == '1' and rank_carts < self.max_cands:
                    cands.append([aid_str, int(reco[0][:-2]),
                                 f'w2v_{self.window_str}_carts', rank_carts])
                    rank_carts += 1
                elif rank_orders < self.max_cands:
                    cands.append([aid_str, int(reco[0][:-2]),
                                 f'w2v_{self.window_str}_orders', rank_orders])
                    rank_orders += 1

        cands = pl.DataFrame(cands, orient='row', columns=[
            'aid_str', 'aid', 'col_name', 'rank'])
        cands = cands.pivot(index=['aid_str', 'aid'],
                            columns='col_name', values='rank')

        if f'w2v_{self.window_str}_clicks' not in cands.columns:
            cands = cands.with_column(pl.lit(None).cast(pl.Int64).alias(
                f'w2v_{self.window_str}_clicks'))

        if f'w2v_{self.window_str}_carts' not in cands.columns:
            cands = cands.with_column(
                pl.lit(None).cast(pl.Int64).alias(f'w2v_{self.window_str}_carts'))

        if f'w2v_{self.window_str}_orders' not in cands.columns:
            cands = cands.with_column(pl.lit(None).cast(pl.Int64).alias(
                f'w2v_{self.window_str}_orders'))

        columns = cands.columns
        columns.sort()

        cands = cands.select(pl.col(columns))

        return cands


class Covisit(CandiadateGen):
    def __init__(self, fold, name, data_path, max_cands, type_weight, days_back, before_time, after_time, time_weight_coef=3, normalize=True, session_hist=30):
        super().__init__(fold=fold, name=name, data_path=data_path)
        self.fold = fold
        self.max_cands = max_cands
        self.maxs = {'': 1661723999,
                     'valid3__': 1661119199,
                     'valid2__': 1660514399,
                     'valid1__': 1659909599}

        self.type_weight = type_weight
        self.days_back = days_back
        self.session_hist = session_hist
        self.before_time = before_time  # positive number
        self.after_time = after_time
        self.normalize = normalize
        self.time_weight_coef = time_weight_coef

    def prepare_candidates(self):

        max_ts = self.maxs[self.fold]
        min_ts = max_ts - (24 * 60 * 60 * self.days_back)

        df1 = pl.scan_parquet(
            f'{self.data_path}raw/{self.fold}test.parquet')
        df = pl.scan_parquet(
            f'{self.data_path}raw/{self.fold}train.parquet')
        df = pl.concat([df, df1])
        del df1

        df = df.filter(pl.col('ts') >= min_ts)

        df = df.sort(by=['session', 'ts'], reverse=[False, True])
        df = df.with_column(
            pl.col('session').cumcount().over('session').alias('n'))
        df = df.filter(pl.col('n') < self.session_hist).drop('n')
        df = df.join(df, on='session')
        df = df.filter(((pl.col('ts_right') - pl.col('ts')) >= - self.before_time) & ((pl.col(
            'ts_right') - pl.col('ts')) <= self.after_time) & (pl.col('aid') != pl.col('aid_right')))
        df = df.with_column(pl.col('type_right').apply(
            lambda x: self.type_weight[x]).alias('wgt'))
        df = df.with_column(pl.col(
            'wgt') * (1 + self.time_weight_coef * ((pl.col('ts') - min_ts) / (max_ts - min_ts))))
        df = df.select(['aid', 'aid_right', 'wgt'])
        df = df.groupby(['aid', 'aid_right']).agg(pl.col('wgt').sum())
        df = df.sort(by=['aid', 'wgt'], reverse=[False, True])
        df = df.with_column(pl.col('aid').cumcount().over('aid').alias('n'))
        if self.normalize:
            aid_wgt_sum = df.groupby('aid').agg(
                pl.col('wgt').sum().alias('wgt_sum'))
            df = df.join(aid_wgt_sum, on='aid')
            df = df.with_column(
                pl.col('wgt') / pl.col('wgt_sum')).drop('wgt_sum')
        df = df.filter(pl.col('n') < self.max_cands).drop('n')

        df = df.collect()

        reco = pl.read_parquet(
            f'{self.data_path}raw/{self.fold}test.parquet')
        reco = reco.sort(by=['session', 'ts'], reverse=[False, True])
        reco = reco.with_column(
            pl.col('session').cumcount().over('session').alias('n'))
        reco = reco.filter(pl.col('n') < self.session_hist).drop('n')
        reco = reco.join(df, on='aid')
        reco = reco.groupby(['session', 'aid_right']).agg(pl.col('wgt').sum())
        reco = reco.sort(by=['session', 'wgt'], reverse=[False, True])
        reco = reco.with_column(
            pl.col('session').cumcount().over('session').alias(self.name))
        reco = reco.filter(pl.col(self.name) < self.max_cands).drop('wgt')
        reco.columns = ['session', 'aid', self.name]

        reco.write_parquet(
            f'{self.data_path}candidates/{self.fold}{self.name}.parquet')


class CovisitMaster(CandiadateGen):
    def __init__(self, fold, name, data_path, max_cands, type_weight, days_back, before_time, after_time, left_types, right_types,
                 time_weight_coef=3, normalize=True, session_hist=30, weekdays=None, dayparts=None):
        super().__init__(fold=fold, name=name, data_path=data_path)
        self.fold = fold
        self.max_cands = max_cands
        self.maxs = {'': 1662328791,
                     'valid3__': 1661723998,
                     'valid2__': 1661119195,
                     'valid1__': 1660514389}

        self.type_weight = type_weight
        self.days_back = days_back
        self.session_hist = session_hist
        self.before_time = before_time  # positive number
        self.after_time = after_time
        self.normalize = normalize
        self.time_weight_coef = time_weight_coef
        self.left_types = left_types
        self.right_types = right_types
        self.weekdays = weekdays
        self.dayparts = dayparts

    def prepare_candidates(self, return_df=False):

        max_ts = self.maxs[self.fold]
        min_ts = max_ts - (24 * 60 * 60 * self.days_back)

        df = pl.read_parquet(
            f'../data/raw/{self.fold}test.parquet')
        df1 = pl.read_parquet(
            f'../data/raw/{self.fold}train.parquet')
        df = pl.concat([df, df1])
        del df1

        df = df.filter(pl.col('ts') >= min_ts)

        if self.weekdays is not None:
            df = df.with_columns(
                (pl.col('ts').cast(pl.Int64) *
                 1000000).cast(pl.Datetime).dt.weekday().alias('weekday'))
            df = df.filter(pl.col('weekday').is_in(
                self.weekdays)).drop('weekday')

        if self.dayparts is not None:
            df = df.with_columns(
                ((pl.col('ts').cast(pl.Int64) *
                  1000000).cast(pl.Datetime).dt.hour() / 6).cast(pl.UInt8).alias('daypart'))
            df = df.filter(pl.col('daypart').is_in(
                self.dayparts)).drop('daypart')

        df = df.sort(by=['session', 'ts'], reverse=[False, True])
        df = df.with_column(
            pl.col('session').cumcount().over('session').alias('n'))
        df = df.filter(pl.col('n') < self.session_hist).drop('n')
        df = df.join(df, on='session')
        df = df.filter(((pl.col('ts_right') - pl.col('ts')) >= - self.before_time) & ((pl.col(
            'ts_right') - pl.col('ts')) <= self.after_time) & (pl.col('aid') != pl.col('aid_right')))
        df = df.filter(pl.col('type').is_in(self.left_types) &
                       pl.col('type_right').is_in(self.right_types))
        df = df.with_column(pl.col('type_right').apply(
            lambda x: self.type_weight[x]).alias('wgt'))
        df = df.with_column(pl.col('wgt') * (1 + self.time_weight_coef *
                            ((pl.col('ts') - min_ts) / (max_ts - min_ts))))
        df = df.select(['aid', 'aid_right', 'wgt'])
        df = df.groupby(['aid', 'aid_right']).agg(pl.col('wgt').sum())
        df = df.sort(by=['aid', 'wgt'], reverse=[False, True])
        df = df.with_column(pl.col('aid').cumcount().over('aid').alias('n'))
        if self.normalize:
            aid_wgt_sum = df.groupby('aid').agg(
                pl.col('wgt').sum().alias('wgt_sum'))
            df = df.join(aid_wgt_sum, on='aid')
            df = df.with_column(
                pl.col('wgt') / pl.col('wgt_sum')).drop('wgt_sum')
        df = df.filter(pl.col('n') < self.max_cands).drop('n')
        reco = pl.read_parquet(
            f'../data/raw/{self.fold}test.parquet')

        if self.weekdays is not None:
            reco = reco.with_columns(
                (pl.col('ts').cast(pl.Int64) *
                 1000000).cast(pl.Datetime).dt.weekday().alias('weekday'))
            reco = reco.filter(pl.col('weekday').is_in(
                self.weekdays)).drop('weekday')

        if self.dayparts is not None:
            reco = reco.with_columns(
                ((pl.col('ts').cast(pl.Int64) *
                  1000000).cast(pl.Datetime).dt.hour() / 6).cast(pl.UInt8).alias('daypart'))
            reco = reco.filter(pl.col('daypart').is_in(
                self.dayparts)).drop('daypart')
        reco = reco.sort(by=['session', 'ts'], reverse=[False, True])
        reco = reco.with_column(
            pl.col('session').cumcount().over('session').alias('n'))
        reco = reco.filter(pl.col('n') < self.session_hist).drop('n')
        reco = reco.filter(pl.col('type').is_in(self.left_types))
        reco = reco.join(df, on='aid')
        reco = reco.groupby(['session', 'aid_right']).agg(pl.col('wgt').sum())
        reco = reco.sort(by=['session', 'wgt'], reverse=[False, True])
        reco = reco.with_column(
            pl.col('session').cumcount().over('session').alias('rank'))
        reco = reco.filter(pl.col('rank') < self.max_cands).drop('wgt')
        reco.columns = ['session', 'aid', self.name]

        if return_df:
            return reco
        else:
            reco.write_parquet(
                f'{self.data_path}candidates/{self.fold}{self.name}.parquet')
