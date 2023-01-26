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
import joblib

COMPUTE_BPR = False


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

    w2v = W2VMaster(fold=fold, name='w2v_cands',
                    data_path='../data/', max_cands=30, session_hist=10)
    w2v = w2v.load_candidates_file()

    candidates = candidates.join(
        w2v, on=['session', 'aid'], how='outer')
    del w2v

    if COMPUTE_BPR:
        candidates = candidates.select(pl.col(['session', 'aid']))

    ses2ses3 = Session2Session(fold=fold, name='ses2ses3', data_path='../data/', max_cands=30,
                               type_weight={0: 1, 1: 6, 2: 3}, days_back=7, types=[1], session_hist=30)
    ses2ses3 = ses2ses3.load_candidates_file(max_rank=30)

    candidates = candidates.join(
        ses2ses3, on=['session', 'aid'], how='outer')
    del ses2ses3

    ses2ses4 = Session2Session(fold=fold, name='ses2ses4', data_path='../data/', max_cands=30,
                               type_weight={0: 1, 1: 6, 2: 3}, days_back=7, types=[2], session_hist=30)
    ses2ses4 = ses2ses4.load_candidates_file(max_rank=30)

    candidates = candidates.join(
        ses2ses4, on=['session', 'aid'], how='outer')
    del ses2ses4

    ses2ses5 = Session2Session(fold=fold, name='ses2ses5', data_path='../data/', max_cands=30,
                               type_weight={0: 1, 1: 6, 2: 3}, days_back=7, types=[1, 2], session_hist=30)
    ses2ses5 = ses2ses5.load_candidates_file(max_rank=30)

    candidates = candidates.join(
        ses2ses5, on=['session', 'aid'], how='outer')
    del ses2ses5

    if COMPUTE_BPR:
        candidates = candidates.select(pl.col(['session', 'aid']))

    print(1, candidates.shape)

    recbole_clicks = CandsFromSubmission(
        fold=fold, event_type_str='clicks', name='recbole_clicks', data_path=config.data_path, base_file_name='recbole', reverse=False)
    recbole_clicks = recbole_clicks.load_candidates_file()

    candidates = candidates.join(
        recbole_clicks, on=['session', 'aid'], how='outer')
    del recbole_clicks

    if COMPUTE_BPR:
        candidates = candidates.select(pl.col(['session', 'aid']))

    recbole_clicks2 = CandsFromSubmission(
        fold=fold, event_type_str='clicks', name='recbole_clicks2', data_path=config.data_path, base_file_name='recbole2', reverse=False)
    recbole_clicks2 = recbole_clicks2.load_candidates_file()

    candidates = candidates.join(
        recbole_clicks2, on=['session', 'aid'], how='outer')
    del recbole_clicks2

    if COMPUTE_BPR:
        candidates = candidates.select(pl.col(['session', 'aid']))

    print(2, candidates.shape)

    covisit1_clicks = CandsFromSubmission(
        fold=fold, event_type_str='clicks', name='covisit1_clicks', data_path=config.data_path, base_file_name='covisit1', reverse=False)
    covisit1_clicks = covisit1_clicks.load_candidates_file()

    candidates = candidates.join(
        covisit1_clicks, on=['session', 'aid'], how='outer')
    del covisit1_clicks

    if COMPUTE_BPR:
        candidates = candidates.select(pl.col(['session', 'aid']))

    covisit1_carts = CandsFromSubmission(
        fold=fold, event_type_str='carts', name='covisit1_carts', data_path=config.data_path, base_file_name='covisit1', reverse=False)
    covisit1_carts = covisit1_carts.load_candidates_file()

    candidates = candidates.join(
        covisit1_carts, on=['session', 'aid'], how='outer')
    del covisit1_carts

    if COMPUTE_BPR:
        candidates = candidates.select(pl.col(['session', 'aid']))

    print(3, candidates.shape)

    mf1_clicks = CandsFromSubmission(
        fold=fold, event_type_str='clicks', name='mf1_clicks', data_path=config.data_path, base_file_name='matrix_factorization1', reverse=False)
    mf1_clicks = mf1_clicks.load_candidates_file()

    candidates = candidates.join(
        mf1_clicks, on=['session', 'aid'], how='outer')
    del mf1_clicks

    if COMPUTE_BPR:
        candidates = candidates.select(pl.col(['session', 'aid']))

    mf1_carts = CandsFromSubmission(
        fold=fold, event_type_str='carts', name='mf1_carts', data_path=config.data_path, base_file_name='matrix_factorization1', reverse=False)
    mf1_carts = mf1_carts.load_candidates_file()

    candidates = candidates.join(
        mf1_carts, on=['session', 'aid'], how='outer')
    del mf1_carts

    if COMPUTE_BPR:
        candidates = candidates.select(pl.col(['session', 'aid']))

    mf1_orders = CandsFromSubmission(
        fold=fold, event_type_str='orders', name='mf1_orders', data_path=config.data_path, base_file_name='matrix_factorization1', reverse=False)
    mf1_orders = mf1_orders.load_candidates_file()

    candidates = candidates.join(
        mf1_orders, on=['session', 'aid'], how='outer')
    del mf1_orders

    if COMPUTE_BPR:
        candidates = candidates.select(pl.col(['session', 'aid']))

    print(4, candidates.shape)

    w2v_window09 = W2VReco(
        fold, 'w2v_window09', config.data_path, '09', 30)
    w2v_window09 = w2v_window09.load_candidates_file(max_rank=10)

    candidates = candidates.join(
        w2v_window09, on=['session', 'aid'], how='outer')
    del w2v_window09

    if COMPUTE_BPR:
        candidates = candidates.select(pl.col(['session', 'aid']))

    w2v_window01 = W2VReco(
        fold, 'w2v_window01', config.data_path, '01', 30)
    w2v_window01 = w2v_window01.load_candidates_file(max_rank=10)

    candidates = candidates.join(
        w2v_window01, on=['session', 'aid'], how='outer')
    del w2v_window01

    if COMPUTE_BPR:
        candidates = candidates.select(pl.col(['session', 'aid']))

    w2v_window35 = W2VReco(
        fold, 'w2v_window35', config.data_path, '35', 30)
    w2v_window35 = w2v_window35.load_candidates_file(max_rank=10)

    candidates = candidates.join(
        w2v_window35, on=['session', 'aid'], how='outer')
    del w2v_window35

    if COMPUTE_BPR:
        candidates = candidates.select(pl.col(['session', 'aid']))

    print(5, candidates.shape)

    covisit2 = Covisit(fold=fold, name='civisit2', data_path='../data/', max_cands=100, type_weight={0: 1, 1: 6, 2: 3},
                       days_back=14, before_time=0, after_time=24 * 60 * 60, )
    covisit2 = covisit2.load_candidates_file(max_rank=100)

    candidates = candidates.join(
        covisit2, on=['session', 'aid'], how='outer')
    del covisit2
    print(5.1, candidates.shape)

    if COMPUTE_BPR:
        candidates = candidates.select(pl.col(['session', 'aid']))

    covisit3 = CovisitMaster(fold=fold, name='civisit3', data_path='../data/', max_cands=30, type_weight={0: 1, 1: 6, 2: 3},
                             days_back=14, before_time=0, after_time=24 * 60 * 60, left_types=[2], right_types=[2])
    covisit3 = covisit3.load_candidates_file(max_rank=20)

    candidates = candidates.join(
        covisit3, on=['session', 'aid'], how='outer')
    del covisit3
    print(5.2, candidates.shape)

    if COMPUTE_BPR:
        candidates = candidates.select(pl.col(['session', 'aid']))

    covisit4 = CovisitMaster(fold=fold, name='civisit4', data_path='../data/', max_cands=100, type_weight={0: 1, 1: 6, 2: 3},
                             days_back=14, before_time=0, after_time=24 * 60 * 60, left_types=[1], right_types=[0])
    covisit4 = covisit4.load_candidates_file(max_rank=100)

    candidates = candidates.join(
        covisit4, on=['session', 'aid'], how='outer')
    del covisit4
    print(5.3, candidates.shape)

    if COMPUTE_BPR:
        candidates = candidates.select(pl.col(['session', 'aid']))

    covisit5 = CovisitMaster(fold=fold, name='civisit5', data_path='../data/', max_cands=100, type_weight={0: 1, 1: 6, 2: 3},
                             days_back=14, before_time=0, after_time=24 * 60 * 60, left_types=[1], right_types=[1])
    covisit5 = covisit5.load_candidates_file(max_rank=100)

    candidates = candidates.join(
        covisit5, on=['session', 'aid'], how='outer')
    del covisit5
    print(5.4, candidates.shape)

    if COMPUTE_BPR:
        candidates = candidates.select(pl.col(['session', 'aid']))

    covisit6 = CovisitMaster(fold=fold, name='covisit6', data_path='../data/', max_cands=100, type_weight={0: 1, 1: 6, 2: 3},
                             days_back=14, before_time=0, after_time=24 * 60 * 60, left_types=[1, 2], right_types=[1, 2])
    covisit6 = covisit6.load_candidates_file(max_rank=100)

    candidates = candidates.join(
        covisit6, on=['session', 'aid'], how='outer')
    del covisit6
    print(5.5, candidates.shape)

    if COMPUTE_BPR:
        candidates = candidates.select(pl.col(['session', 'aid']))

    covisit7 = CovisitMaster(fold=fold, name='covisit7', data_path='../data/', max_cands=100, type_weight={0: 1, 1: 17, 2: 42},
                             days_back=7, before_time=4 * 60 * 60, after_time=24 * 60 * 60, left_types=[1], right_types=[1], time_weight_coef=0.15, session_hist=24)
    covisit7 = covisit7.load_candidates_file(max_rank=100)

    candidates = candidates.join(
        covisit7, on=['session', 'aid'], how='outer')
    del covisit7
    print(5.6, candidates.shape)

    if COMPUTE_BPR:
        candidates = candidates.select(pl.col(['session', 'aid']))

    covisit8 = CovisitMaster(fold=fold, name='covisit8', data_path='../data/', max_cands=100, type_weight={0: 1, 1: 6, 2: 3},
                             days_back=14, before_time=0, after_time=24 * 60 * 60, left_types=[0, 1, 2], right_types=[0, 1, 2], reco_hist=1)
    covisit8 = covisit8.load_candidates_file(max_rank=100)

    candidates = candidates.join(
        covisit8, on=['session', 'aid'], how='outer')
    del covisit8
    print(5.7, candidates.shape)

    if COMPUTE_BPR:
        candidates = candidates.select(pl.col(['session', 'aid']))

    covisit9 = CovisitMaster(fold=fold, name='covisit9', data_path='../data/', max_cands=30, type_weight={0: 1, 1: 6, 2: 3},
                             days_back=14, before_time=0, after_time=24 * 60 * 60, left_types=[2], right_types=[2], reco_hist=1)
    covisit9 = covisit9.load_candidates_file(max_rank=20)

    candidates = candidates.join(
        covisit9, on=['session', 'aid'], how='outer')
    del covisit9
    print(5.8, candidates.shape)

    if COMPUTE_BPR:
        candidates = candidates.select(pl.col(['session', 'aid']))

    covisit10 = CovisitMaster(fold=fold, name='covisit10', data_path='../data/', max_cands=30, type_weight={0: 1, 1: 6, 2: 3},
                              days_back=14, before_time=0, after_time=24 * 60 * 60, left_types=[1], right_types=[0], reco_hist=1)
    covisit10 = covisit10.load_candidates_file(max_rank=20)

    candidates = candidates.join(
        covisit10, on=['session', 'aid'], how='outer')
    del covisit10
    print(5.9, candidates.shape)

    if COMPUTE_BPR:
        candidates = candidates.select(pl.col(['session', 'aid']))

    covisit11 = CovisitMaster(fold=fold, name='covisit11', data_path='../data/', max_cands=30, type_weight={0: 1, 1: 6, 2: 3},
                              days_back=14, before_time=0, after_time=24 * 60 * 60, left_types=[1], right_types=[1], reco_hist=1)
    covisit11 = covisit11.load_candidates_file(max_rank=20)

    candidates = candidates.join(
        covisit11, on=['session', 'aid'], how='outer')
    del covisit11
    print(5.11, candidates.shape)

    if COMPUTE_BPR:
        candidates = candidates.select(pl.col(['session', 'aid']))

    covisit12 = CovisitMaster(fold=fold, name='covisit12', data_path='../data/', max_cands=30, type_weight={0: 1, 1: 6, 2: 3},
                              days_back=14, before_time=0, after_time=24 * 60 * 60, left_types=[1, 2], right_types=[1, 2], reco_hist=1)
    covisit12 = covisit12.load_candidates_file(max_rank=20)

    candidates = candidates.join(
        covisit12, on=['session', 'aid'], how='outer')
    del covisit12
    print(5.12, candidates.shape)

    if COMPUTE_BPR:
        candidates = candidates.select(pl.col(['session', 'aid']))

    covisit13 = CovisitMaster(fold=fold, name='covisit13', data_path='../data/', max_cands=30, type_weight={0: 1, 1: 17, 2: 42},
                              days_back=7, before_time=4 * 60 * 60, after_time=24 * 60 * 60, left_types=[1], right_types=[1], time_weight_coef=0.15, session_hist=24, reco_hist=1)
    covisit13 = covisit13.load_candidates_file(max_rank=20)

    candidates = candidates.join(
        covisit13, on=['session', 'aid'], how='outer')
    del covisit13

    print(6, candidates.shape)

    if COMPUTE_BPR:
        candidates = candidates.select(pl.col(['session', 'aid']))

    covisit14 = CovisitMaster(fold=fold, name='covisit14', data_path='../data/', max_cands=100, type_weight={0: 1, 1: 6, 2: 3},
                              days_back=14, before_time=-24 * 60 * 60, after_time=7 * 24 * 60 * 60, left_types=[0, 1, 2], right_types=[0, 1, 2], reco_hist=1)
    covisit14 = covisit14.load_candidates_file(max_rank=100)

    candidates = candidates.join(
        covisit14, on=['session', 'aid'], how='outer')
    del covisit14

    if COMPUTE_BPR:
        candidates = candidates.select(pl.col(['session', 'aid']))

    tg_covisit1 = TimeGroupCovisitMaster(fold=fold, name='tg_covisit1', data_path='../data/', max_cands=100, type_weight={0: 1, 1: 6, 2: 3},
                                         days_back=14, before_time=0, after_time=24 * 60 * 60, left_types=[0, 1, 2], right_types=[0, 1, 2])
    tg_covisit1 = tg_covisit1.load_candidates_file(max_rank=100)

    candidates = candidates.join(
        tg_covisit1, on=['session', 'aid'], how='outer')
    del tg_covisit1

    if COMPUTE_BPR:
        candidates = candidates.select(pl.col(['session', 'aid']))

    print(7, candidates.shape)

    cleora_cands = CleoraCands(
        fold=fold, name='cleora_emde', data_path='../data/')
    cleora_cands = cleora_cands.load_candidates_file(max_rank=100)

    candidates = candidates.join(
        cleora_cands, on=['session', 'aid'], how='outer')
    del cleora_cands

    print(8, candidates.shape)

    bpr_reco = BPRReco(fold=fold, name='bpr_cands',
                       data_path='../data/', max_cands=50)
    bpr_reco = bpr_reco.load_candidates_file(max_rank=50)
    candidates = candidates.join(
        bpr_reco, on=['session', 'aid'], how='outer')
    del bpr_reco

    print(9.1, candidates.shape)

    test_sessions = pl.read_parquet(f'../data/raw/{fold}test.parquet')
    test_sessions = test_sessions.select('session').unique()
    candidates = candidates.join(
        test_sessions, on='session', how='inner')

    if COMPUTE_BPR:
        candidates = candidates.select(pl.col(['session', 'aid']))

    print(9.2, candidates.shape)

    tl1 = TopLeak(fold=fold, event_type=0, name='leak_top_day_clicks',
                  data_path='../data/', next_days=False, max_cands=50)
    tl1 = tl1.load_candidates_file(max_rank=10)
    candidates = candidates.join(
        tl1, on=['session', 'aid'], how='outer')
    del tl1

    if COMPUTE_BPR:
        candidates = candidates.select(pl.col(['session', 'aid']))

    tl2 = TopLeak(fold=fold, event_type=1, name='leak_top_day_carts',
                  data_path='../data/', next_days=False, max_cands=50)
    tl2 = tl2.load_candidates_file(max_rank=10)
    candidates = candidates.join(
        tl2, on=['session', 'aid'], how='outer')
    del tl2

    if COMPUTE_BPR:
        candidates = candidates.select(pl.col(['session', 'aid']))

    print(10, candidates.shape)

    tl3 = TopLeak(fold=fold, event_type=2, name='leak_top_day_orders',
                  data_path='../data/', next_days=False, max_cands=50)
    tl3 = tl3.load_candidates_file(max_rank=10)
    candidates = candidates.join(
        tl3, on=['session', 'aid'], how='outer')
    del tl3

    if COMPUTE_BPR:
        candidates = candidates.select(pl.col(['session', 'aid']))

    tl4 = TopLeak(fold=fold, event_type=0, name='leak_top_days_clicks',
                  data_path='../data/', next_days=True, max_cands=50)
    tl4 = tl4.load_candidates_file(max_rank=10)
    candidates = candidates.join(
        tl4, on=['session', 'aid'], how='outer')
    del tl4

    if COMPUTE_BPR:
        candidates = candidates.select(pl.col(['session', 'aid']))

    tl5 = TopLeak(fold=fold, event_type=1, name='leak_top_days_carts',
                  data_path='../data/', next_days=True, max_cands=50)
    tl5 = tl5.load_candidates_file(max_rank=10)
    candidates = candidates.join(
        tl5, on=['session', 'aid'], how='outer')
    del tl5

    if COMPUTE_BPR:
        candidates = candidates.select(pl.col(['session', 'aid']))

    print(11, candidates.shape)

    tl6 = TopLeak(fold=fold, event_type=2, name='leak_top_days_orders',
                  data_path='../data/', next_days=True, max_cands=50)
    tl6 = tl6.load_candidates_file(max_rank=10)
    candidates = candidates.join(
        tl6, on=['session', 'aid'], how='outer')
    del tl6

    if COMPUTE_BPR:
        candidates = candidates.select(pl.col(['session', 'aid']))

    print(12, candidates.shape)

    candidates = candidates.join(
        test_sessions, on='session', how='inner')

    del test_sessions
    print(13, candidates.shape)

    candidates = candidates.fill_null(999)

    cands_cols = candidates.columns
    cands_cols.remove('aid')
    cands_cols.remove('session')

    for col in [
        'leak_top_day_clicks_cnt',
        'leak_top_day_carts_cnt',
        'leak_top_day_orders_cnt',

        'leak_top_days_clicks_cnt',
        'leak_top_days_carts_cnt',
            'leak_top_days_orders_cnt']:
        if col in cands_cols:
            cands_cols.remove(col)
            candidates = candidates.with_columns(pl.col(col).cast(pl.Float32))

    candidates = candidates.with_columns(pl.col(cands_cols).cast(pl.UInt16))

    if COMPUTE_BPR:
        candidates = candidates.select(pl.col(['session', 'aid']))
    print('candidates shape', candidates.shape)

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


class CleoraCands(CandiadateGen):
    def __init__(self, fold, name, data_path, reverse=False):
        super().__init__(fold=fold, name=name, data_path=data_path)
        self.fold = fold
        self.reverse = reverse

    def prepare_candidates(self):
        pass


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
        df = df.drop_nulls()
        df.write_parquet(
            f'{self.data_path}candidates/{self.fold}{self.name}.parquet')


class W2VReco(CandiadateGen):
    def __init__(self, fold, name, data_path, window_str, max_cands, use_event=0):
        super().__init__(fold=fold, name=name, data_path=data_path)
        self.fold = fold
        self.window_str = window_str
        self.max_cands = max_cands
        self.use_event = use_event

    def prepare_candidates(self):
        df = pl.read_parquet(
            f'{self.data_path}raw/{self.fold}test.parquet').lazy()
        model = Word2Vec.load(
            f'{self.data_path}raw/word2vec_w{self.window_str}.model')

        df = df.unique(subset=['session', 'aid', 'type'], keep='last')
        df = df.with_column(pl.col('ts').cumcount(
            reverse=True).over('session').alias('rank'))
        df = df.filter(pl.col('rank') == self.use_event).drop('rank')

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

        if f'{self.name}_clicks' not in cands.columns:
            cands = cands.with_column(pl.lit(None).cast(pl.Int64).alias(
                f'{self.name}_clicks'))

        if f'{self.name}_carts' not in cands.columns:
            cands = cands.with_column(
                pl.lit(None).cast(pl.Int64).alias(f'{self.name}_carts'))

        if f'{self.name}_orders' not in cands.columns:
            cands = cands.with_column(pl.lit(None).cast(pl.Int64).alias(
                f'{self.name}_orders'))

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
                 time_weight_coef=3, normalize=True, session_hist=30, weekdays=None, dayparts=None, reco_hist=30):
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
        self.reco_hist = reco_hist

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
                ((pl.col('ts').cast(pl.Int64) + 7200) *
                 1000000).cast(pl.Datetime).dt.weekday().alias('weekday'))
            df = df.filter(pl.col('weekday').is_in(
                self.weekdays)).drop('weekday')

        if self.dayparts is not None:
            df = df.with_columns((
                ((((pl.col('ts').cast(pl.Int64) + 7200) *
                   1000000).cast(pl.Datetime).dt.hour() + 2) / 6) % 4).cast(pl.UInt8).alias('daypart'))
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
                ((pl.col('ts').cast(pl.Int64) + 7200) *
                 1000000).cast(pl.Datetime).dt.weekday().alias('weekday'))
            reco = reco.filter(pl.col('weekday').is_in(
                self.weekdays)).drop('weekday')

        if self.dayparts is not None:
            reco = reco.with_columns((
                ((((pl.col('ts').cast(pl.Int64) + 7200) *
                   1000000).cast(pl.Datetime).dt.hour() + 2) / 6) % 4).cast(pl.UInt8).alias('daypart'))
            reco = reco.filter(pl.col('daypart').is_in(
                self.dayparts)).drop('daypart')
        reco = reco.sort(by=['session', 'ts'], reverse=[False, True])
        reco = reco.with_column(
            pl.col('session').cumcount().over('session').alias('n'))
        reco = reco.filter(pl.col('n') < self.reco_hist).drop('n')
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


class TimeGroupCovisitMaster(CandiadateGen):
    def __init__(self, fold, name, data_path, max_cands, type_weight, days_back, before_time, after_time, left_types, right_types,
                 time_weight_coef=3, normalize=True, session_hist=30):
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

        df = df.with_columns(
            ((pl.col('ts').cast(pl.Int64) + 7200) *
                1000000).cast(pl.Datetime).dt.weekday().alias('weekday'))

        df = df.with_columns((
            ((((pl.col('ts').cast(pl.Int64) + 7200) *
               1000000).cast(pl.Datetime).dt.hour() + 2) / 6) % 4).cast(pl.UInt8).alias('daypart'))

        df = df.with_column(
            (pl.col('weekday') * 10 + pl.col('daypart')).alias('time_group'))

        df = df.sort(by=['session', 'ts'], reverse=[False, True])
        df = df.with_column(
            pl.col('session').cumcount().over('session').alias('n'))
        df = df.filter(pl.col('n') < self.session_hist).drop('n')
        df = df.join(df, on='session')
        df = df.filter(((pl.col('ts_right') - pl.col('ts')) >= - self.before_time) & ((pl.col(
            'ts_right') - pl.col('ts')) <= self.after_time) & (pl.col('aid') != pl.col('aid_right')))
        df = df.filter(pl.col('type').is_in(self.left_types) &
                       pl.col('type_right').is_in(self.right_types))
        df = df.filter(pl.col('time_group') <= pl.col('time_group_right'))
        df = df.with_column(pl.col('type_right').apply(
            lambda x: self.type_weight[x]).alias('wgt'))
        df = df.with_column(pl.col('wgt') * (1 + self.time_weight_coef *
                            ((pl.col('ts') - min_ts) / (max_ts - min_ts))))
        df = df.select(['time_group', 'aid', 'aid_right', 'wgt'])
        df = df.groupby(['time_group', 'aid', 'aid_right']
                        ).agg(pl.col('wgt').sum())
        df = df.sort(by=['time_group', 'aid', 'wgt'],
                     reverse=[False, False, True])
        df = df.with_column(pl.col(['time_group', 'aid']).cumcount().over(
            ['time_group', 'aid']).alias('n'))
        if self.normalize:
            aid_wgt_sum = df.groupby(['time_group', 'aid']).agg(
                pl.col('wgt').sum().alias('wgt_sum'))
            df = df.join(aid_wgt_sum, on=['time_group', 'aid'])
            df = df.with_column(
                pl.col('wgt') / pl.col('wgt_sum')).drop('wgt_sum')
        df = df.filter(pl.col('n') < self.max_cands).drop('n')
        reco = pl.read_parquet(
            f'../data/raw/{self.fold}test.parquet')

        reco = reco.with_columns(
            ((pl.col('ts').cast(pl.Int64) + 7200) *
                1000000).cast(pl.Datetime).dt.weekday().alias('weekday'))

        reco = reco.with_columns((
            ((((pl.col('ts').cast(pl.Int64) + 7200) *
               1000000).cast(pl.Datetime).dt.hour() + 2) / 6) % 4).cast(pl.UInt8).alias('daypart'))

        reco = reco.with_column(
            (pl.col('weekday') * 10 + pl.col('daypart')).alias('time_group'))
        max_time_group = reco.select(['time_group', 'session']).groupby(
            'session').agg(pl.col('time_group').max().alias('max_time_group'))
        reco = reco.join(max_time_group, on='session', how='inner')
        del max_time_group
        reco = reco.sort(by=['session', 'ts'], reverse=[False, True])
        reco = reco.with_column(
            pl.col('session').cumcount().over('session').alias('n'))
        reco = reco.filter(pl.col('n') < self.session_hist).drop('n')
        reco = reco.filter(pl.col('type').is_in(self.left_types))
        reco = reco.filter(pl.col('time_group') >= pl.col(
            'max_time_group') - 1).drop('max_time_group')
        reco = reco.join(df, on=['time_group', 'aid'])
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


class Session2Session(CandiadateGen):

    def __init__(self, fold, name, data_path, max_cands, type_weight, days_back, types, session_hist=30):
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
        self.types = types

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

        df = df.filter(pl.col('type').is_in(self.types))
        df = df.sort(by=['session', 'ts'], reverse=[False, True])
        df = df.with_column(
            pl.col('session').cumcount().over('session').alias('n'))
        df = df.filter(pl.col('n') < self.session_hist).drop('n')
        df = df.drop('ts')
        df2 = df.join(df, on='aid')
        df2 = df2.drop('type_right')
        df = df2.join(df, left_on='session_right', right_on='session')
        del df2
        df = df.with_column(pl.col('type_right').apply(
            lambda x: self.type_weight[x]).alias('wgt'))
        df = df.groupby(pl.col(['session', 'aid_right'])).agg(pl.sum('wgt'))
        df = df.sort(by=['session', 'wgt'], reverse=[False, True])
        df = df.with_column(
            pl.col('session').cumcount().over('session').alias('rank'))
        df = df.filter(pl.col('rank') < self.max_cands).drop('wgt')
        df.columns = ['session', 'aid', self.name]

        if return_df:
            return df
        else:
            df.write_parquet(
                f'{self.data_path}candidates/{self.fold}{self.name}.parquet')


class W2VMaster(CandiadateGen):

    def __init__(self, fold, name, data_path, max_cands, session_hist=10):
        super().__init__(fold=fold, name=name, data_path=data_path)
        self.fold = fold
        self.name = name
        self.data_path = data_path
        self.max_cands = max_cands
        self.session_hist = session_hist

    def prepare_candidates(self, return_df=False):

        model = Word2Vec.load(f'{self.data_path}raw/{self.fold}word2vec.model')
        vocab = list(set([int(i) for i in model.wv.key_to_index.keys()]))
        df = pl.read_parquet(f'{self.data_path}raw/{self.fold}test.parquet')
        df = df.unique(subset=['session', 'aid'], keep='last')
        df = df.filter(pl.col('aid').is_in(vocab))
        df = df.with_column(pl.col('session').cumcount(
            reverse=True).over('session').alias('rank'))
        df = df.filter(pl.col('rank') < self.session_hist)
        df = df.with_column((0.9 ** pl.col('rank')).alias('wgt'))
        df = df.drop(['ts', 'type', 'rank'])
        session = df.groupby('session').agg(
            pl.col('wgt').sum().alias('wgt_sum'))
        df = df.join(session, on='session')
        del session
        df = df.with_column(pl.col('wgt') / pl.col('wgt_sum')).drop('wgt_sum')
        df_w2v = pl.DataFrame(model.wv.vectors)
        df_w2v = df_w2v.select(pl.col('*').cast(pl.Float32))
        aids = pl.DataFrame(
            [int(i) for i in model.wv.key_to_index.keys()], columns=['aid'])
        df_w2v = pl.concat([aids, df_w2v], how='horizontal')
        df_w2v = df_w2v.with_column(pl.col('aid').cast(pl.Int32))
        df = df.join(df_w2v, on='aid')
        df = df.with_columns(
            pl.col([f'column_{i}' for i in range(100)]) / pl.col('wgt'))
        df = df.drop(['aid', 'wgt'])
        df = df.groupby('session').sum()
        annoy_index = AnnoyIndexer(model, 100)
        df = df.to_pandas()
        df.set_index('session', inplace=True)
        recos = []
        sessions = []
        ranks = []
        rank = [i for i in range(self.max_cands)]
        n = df.shape[0]
        for sess, vec in tqdm(df.iloc[:n, :].iterrows(), total=n):
            recos.extend([int(i[0]) for i in model.wv.most_similar(
                [vec.values], topn=self.max_cands, indexer=annoy_index)])
            sessions.extend([sess] * self.max_cands)
            ranks.extend(rank)

        df = pl.DataFrame([sessions, recos, ranks], columns=[
                          'session', 'aid', self.name])
        df = df.select(pl.col('*').cast(pl.Int32))

        if return_df:
            return df
        else:
            df.write_parquet(
                f'{self.data_path}candidates/{self.fold}{self.name}.parquet')


class TopLeak(CandiadateGen):

    def __init__(self, fold, event_type, name, data_path, next_days, max_cands):
        super().__init__(fold=fold, name=name, data_path=data_path)
        self.fold = fold
        self.event_type = event_type
        self.next_days = next_days
        self.max_cands = max_cands

    def prepare_candidates(self, return_df=False):
        df = pl.read_parquet(f'{self.data_path}raw/{self.fold}test.parquet')
        df = df.with_column(
            ((pl.col('ts').cast(pl.Int64) + 7200) *
             1000000).cast(pl.Datetime).dt.weekday().alias('weekday'))
        if self.event_type is not None:
            tops = df.filter(pl.col('type') == self.event_type).groupby(
                ['weekday', 'aid']).count()
        else:
            tops = df.groupby(['weekday', 'aid']).count()
        tops.columns = ['weekday', 'aid', f'cnt']
        tops = tops.sort(by=['weekday', f'cnt'], reverse=True)
        tops_all = []
        for i in range(7):
            if self.next_days:
                tops_part = tops.filter(pl.col('weekday') > i)
            else:
                tops_part = tops.filter(pl.col('weekday') == (i + 1))
            tops_part = tops_part.with_column(
                (pl.col('cnt') * (2 ** (pl.col('weekday') - i - 1))).cast(pl.UInt32))
            tops_part = tops_part.groupby(['aid']).sum().drop('weekday')
            tops_part = tops_part.sort(by='cnt', reverse=True)
            tops_part = tops_part.select(
                [pl.col(['aid', f'cnt']), pl.col('aid').cumcount().alias(f'rank')])
            sum_cnt = tops_part.select(pl.col('cnt')).sum()[0, 0]
            tops_part = tops_part.with_column(pl.col('cnt') / sum_cnt)
            tops_part = tops_part.filter(pl.col('rank') < self.max_cands)
            tops_part = tops_part.with_column(
                pl.lit(i).cast(pl.UInt32).alias('weekday'))
            tops_all.append(tops_part)
        tops_all = pl.concat(tops_all, how='vertical')
        df = df.groupby('session').max().select(['session', 'weekday'])
        df = df.join(tops_all, on='weekday')
        df = df.select(['session', 'aid', 'rank', 'cnt'])
        df.columns = ['session', 'aid', self.name, f'{self.name}_cnt']
        if return_df:
            return df
        else:
            df.write_parquet(
                f'{self.data_path}candidates/{self.fold}{self.name}.parquet')


class BPRReco(CandiadateGen):

    def __init__(self, fold, name, data_path, max_cands):
        super().__init__(fold=fold, name=name, data_path=data_path)
        self.fold = fold
        self.name = name
        self.data_path = data_path
        self.max_cands = max_cands

    def prepare_candidates(self, return_df=False):
        df = pl.read_parquet(f'{self.data_path}raw/{self.fold}test.parquet')
        df = df.with_column(pl.col('type') + 1)
        df = df.select(pl.col(['session', 'aid', 'type'])
                       ).groupby(['session', 'aid']).max()

        session_inv = pl.read_parquet(
            f'{self.data_path}features/{self.fold}bpr_score_session_inv.parquet')
        aid_inv = pl.read_parquet(
            f'{self.data_path}features/{self.fold}bpr_score_aid_inv.parquet')

        df = df.join(session_inv, on='session')
        df = df.join(aid_inv, on='aid')
        values = df.select('type').to_numpy().ravel()
        aid_idx = df.select('aid_idx').to_numpy().ravel()
        session_idx = df.select('session_idx').to_numpy().ravel()
        min_session = session_idx.min()
        sess = np.array([i for i in range(min_session)]).ravel()
        vals = np.array([1 for _ in range(min_session)]).ravel()
        aids = np.array([0 for _ in range(min_session)]).ravel()

        session_idx = np.concatenate([sess, session_idx])
        aid_idx = np.concatenate([aids, aid_idx])
        values = np.concatenate([vals, values])

        aid_session = scipy.sparse.coo_matrix((values, (aid_idx, session_idx)), shape=(aid_idx.max()+1,
                                                                                       session_idx.max()+1))
        aid_session = aid_session.transpose()
        aid_session = aid_session.tocsr()
        model = joblib.load(
            f'{self.data_path}features/{self.fold}bpr_score_model.pkl')
        users = np.unique(session_idx)

        x = model.recommend(userid=users, user_items=aid_session, N=self.max_cands,
                            filter_already_liked_items=True, recalculate_user=False)[0]
        reco = pl.DataFrame([list(np.repeat(users, self.max_cands)), list(
            x.ravel())], columns=['session_idx', 'aid_idx'])
        reco = reco.with_column(
            pl.col('aid_idx').cumcount().over('session_idx').alias('rank'))
        reco = reco.select(
            pl.col(['session_idx', 'aid_idx', 'rank']).cast(pl.Int32))

        reco = reco.join(session_inv, on='session_idx')
        reco = reco.join(aid_inv, on='aid_idx')
        reco = reco.select(['session', 'aid', 'rank'])
        reco.columns = ['session', 'aid', self.name]

        if return_df:
            return reco
        else:
            reco.write_parquet(
                f'{self.data_path}candidates/{self.fold}{self.name}.parquet')
