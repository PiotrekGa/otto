import polars as pl
import numpy as np
import scipy
from pathlib import Path
from tqdm import tqdm
from gensim.models import Word2Vec
from gensim.similarities.annoy import AnnoyIndexer
import implicit


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

    # bpr1 = BPRReco(
    #     fold, 'bpr1', config.data_path, 30)
    # bpr1 = bpr1.load_candidates_file(max_rank=5)

    # candidates = candidates.join(
    #     bpr1, on=['session', 'aid'], how='outer')
    # del bpr1

    candidates = candidates.fill_null(999)

    # columns = candidates.columns
    # columns.remove('session')
    # columns.remove('aid')
    # columns = ['cand__' + i for i in columns]
    # columns = ['session', 'aid'] + columns
    # candidates = candidates.select(columns)

    return candidates


class CandiadateGen():

    def __init__(self, fold, name, data_path):
        self.fold = fold
        self.name = name
        self.data_path = data_path

    def prepare_candidates(self):
        raise NotImplementedError

    @staticmethod
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


class BPRReco(CandiadateGen):

    def __init__(self, fold, name, data_path, max_cands):
        super().__init__(fold=fold, name=name, data_path=data_path)
        self.fold = fold
        self.max_cands = max_cands

    def prepare_candidates(self):
        df = pl.read_parquet(
            f'{self.data_path}raw/{self.fold}test.parquet')
        df = df.select(pl.col(['session', 'aid', 'type'])).unique()
        aid_cnt = df.groupby('aid').agg(
            pl.col('session').n_unique().alias('cnt'))
        aid_cnt = aid_cnt.filter(pl.col('cnt') >= 5)
        df = df.join(aid_cnt, on='aid').drop('cnt')
        df = df.groupby(['session', 'aid']).agg(pl.col('type').max())
        df = df.with_column(
            (pl.col('session').rank('dense') - 1).alias('session_idx'))
        df = df.with_column((pl.col('aid').rank('dense') - 1).alias('aid_idx'))
        values = df.select(pl.col('type')).to_numpy().ravel()
        session_idx = df.select(pl.col('session_idx')).to_numpy().ravel()
        aid_idx = df.select(pl.col('aid_idx')).to_numpy().ravel()
        session_aid = scipy.sparse.coo_matrix((values, (session_idx, aid_idx)), shape=(
            np.unique(session_idx).shape[0], np.unique(aid_idx).shape[0]))
        session_idx = np.unique(session_idx)
        session_aid = session_aid.tocsr()
        model = implicit.bpr.BayesianPersonalizedRanking(64)
        model.fit(session_aid)
        batch_num = 0
        batch_size = 1000
        result = []
        while batch_num * batch_size < session_idx.shape[0]:
            batch_sessions = session_idx[batch_num *
                                         batch_size: (batch_num + 1) * batch_size]
            batch_aids, _ = model.recommend(
                batch_sessions, session_aid[batch_sessions], self.max_cands)
            batch_num += 1
            batch_sessions = np.repeat(
                batch_sessions, self.max_cands).reshape(-1, 1)
            batch_aids = batch_aids.ravel().reshape(-1, 1)
            result.append(pl.DataFrame(
                np.hstack([batch_sessions, batch_aids]), columns=['session_idx', 'aid_idx']))
        result = pl.concat(result)
        result = result.with_column(pl.lit(1).alias('one'))
        result = result.with_column((pl.col('one').cumsum().over(
            'session_idx') - 1).alias('bpr1')).drop('one')
        session_inv = df.select(pl.col(['session', 'session_idx'])).unique()
        aid_inv = df.select(pl.col(['aid', 'aid_idx'])).unique()

        session_inv = session_inv.select(pl.col('*').cast(pl.Int32))
        aid_inv = aid_inv.select(pl.col('*').cast(pl.Int32))
        result = result.select(pl.col('*').cast(pl.Int32))
        result = result.join(session_inv, on='session_idx')
        result = result.join(aid_inv, on='aid_idx')
        result = result.drop(['session_idx', 'aid_idx'])
        result.write_parquet(
            f'{self.data_path}candidates/{self.fold}{self.name}.parquet')
