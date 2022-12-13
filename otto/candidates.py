import polars as pl
import os
from pathlib import Path
from tqdm import tqdm
from gensim.models import Word2Vec
from gensim.similarities.annoy import AnnoyIndexer


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
        values='rank', index=['session', 'aid'], columns='name')

    covisit1_clicks = CandsFromSubmission(
        fold, 'clicks', 'covisit1_clicks', config.data_path, 'covisit1')
    covisit1_clicks = covisit1_clicks.load_candidates_file()

    candidates = candidates.join(
        covisit1_clicks, on=['session', 'aid'], how='outer')
    del covisit1_clicks

    covisit1_carts = CandsFromSubmission(
        fold, 'carts', 'covisit1_carts', config.data_path, 'covisit1')
    covisit1_carts = covisit1_carts.load_candidates_file()

    candidates = candidates.join(
        covisit1_carts, on=['session', 'aid'], how='outer')
    del covisit1_carts

    mf1_clicks = CandsFromSubmission(
        fold, 'clicks', 'mf1_clicks', config.data_path, 'matrix_factorization1')
    mf1_clicks = mf1_clicks.load_candidates_file()

    candidates = candidates.join(
        mf1_clicks, on=['session', 'aid'], how='outer')
    del mf1_clicks

    mf1_carts = CandsFromSubmission(
        fold, 'carts', 'mf1_carts', config.data_path, 'matrix_factorization1')
    mf1_carts = mf1_carts.load_candidates_file()

    candidates = candidates.join(
        mf1_carts, on=['session', 'aid'], how='outer')
    del mf1_carts

    mf1_orders = CandsFromSubmission(
        fold, 'orders', 'mf1_orders', config.data_path, 'matrix_factorization1')
    mf1_orders = mf1_orders.load_candidates_file()

    candidates = candidates.join(
        mf1_orders, on=['session', 'aid'], how='outer')
    del mf1_orders

    w2v_window09 = W2VReco(
        fold, 'w2v_window09', config.data_path, '09', 30)
    w2v_window09 = w2v_window09.load_candidates_file()

    candidates = candidates.join(
        w2v_window09, on=['session', 'aid'], how='outer')
    del w2v_window09

    # w2v_window15 = W2VReco(
    #     fold, 'w2v_window15', config.data_path, '15', 30)
    # w2v_window15 = w2v_window15.load_candidates_file()

    # candidates = candidates.join(
    #     w2v_window15, on=['session', 'aid'], how='outer')
    # del w2v_window15

    candidates = candidates.fill_null(999)
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
            print(f'preparing candidates {self.fold}{self.name}')
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


class CandsFromSubmission(CandiadateGen):

    def __init__(self, fold, event_type_str, name, data_path, base_file_name):
        super().__init__(fold=fold, name=name, data_path=data_path)
        self.fold = fold
        self.event_type_str = event_type_str
        self.base_file_name = base_file_name

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
        df = df.drop(['session_type', 'labels', 'session_type2'])
        df = df.explode('candidates').with_column(
            pl.col('candidates').cumcount().over('session').alias(self.name))
        df = df.filter(pl.col('type_str') == self.event_type_str)
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
        vocab = set(model.wv.index_to_key)
        annoy_index = AnnoyIndexer(model, 100)
        cands = []
        for i in tqdm(range(df.shape[0])):
            cands.extend(self.get_w2v_reco(
                df[i, 4], df[i, 0], model, vocab, annoy_index))
        del df, model, vocab, annoy_index

        if os.path.exists('temp.csv'):
            os.remove('temp.csv')

        with open(f'temp.csv', 'a') as f:
            f.write('session,aid,name,rank\n')
            for i in tqdm(cands):
                x = ','.join([str(j) for j in i]) + '\n'
                f.write(x)
        f.close()

        del cands
        cands = pl.read_csv('temp.csv')

        if os.path.exists('temp.csv'):
            os.remove('temp.csv')

        columns = cands.select(pl.col('name')).unique().to_dict()
        for i, column in enumerate(columns['name']):
            if i == 0:
                df = cands.filter(pl.col('name') == column).with_column(
                    pl.col('rank').alias(column)).drop(['name', 'rank'])
            else:
                df_temp = cands.filter(pl.col('name') == column).with_column(
                    pl.col('rank').alias(column)).drop(['name', 'rank'])
                df = df.join(df_temp, on=['session', 'aid'], how='outer')
        df = df.select(pl.col('*').cast(pl.Int32))
        df.write_parquet(
            f'{self.data_path}candidates/{self.fold}{self.name}.parquet')

    def get_w2v_reco(self, aid_str, session, model, vocab, indexer):
        cands = []
        if aid_str in vocab:
            recos = model.wv.most_similar(aid_str, topn=200, indexer=indexer)
            clicks_rank = 0
            carts_rank = 0
            orders_rank = 0
            for reco in recos:
                if len(reco[0]) > 1:
                    if reco[0][-1] == '0' and clicks_rank < self.max_cands:
                        cands.append(
                            [session, int(reco[0][:-2]), f'w2v_{self.window_str}_clicks', clicks_rank])
                        clicks_rank += 1
                    elif reco[0][-1] == '1' and carts_rank < self.max_cands:
                        cands.append([session, int(reco[0][:-2]),
                                      f'w2v_{self.window_str}_carts', carts_rank])
                        carts_rank += 1
                    elif orders_rank < self.max_cands:
                        cands.append(
                            [session, int(reco[0][:-2]), f'w2v_{self.window_str}_orders', orders_rank])
                        orders_rank += 1
        return cands
