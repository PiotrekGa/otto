import polars as pl
from pathlib import Path
from implicit.nearest_neighbours import bm25_weight
from implicit.bpr import BayesianPersonalizedRanking
import time
import numpy as np
import scipy
import joblib
from tqdm import tqdm
import gc
from candidates import COMPUTE_BPR


def add_labels(candidates, fold, config):
    labels = pl.read_parquet(
        f'{config.data_path}raw/{fold}test_labels.parquet')
    labels = labels.with_column(pl.lit(1).alias('one'))
    labels = labels.pivot(values='one', columns='type',
                          index=['session', 'aid'])
    labels.columns = ['session', 'aid', 'y_clicks', 'y_carts', 'y_orders']
    labels = labels.select([pl.col(['session', 'aid']), pl.col(
        ['y_clicks', 'y_carts', 'y_orders']).cast(pl.UInt8)])
    candidates = candidates.join(labels, on=['session', 'aid'], how='left')
    candidates = candidates.with_columns(
        pl.col(['y_clicks', 'y_carts', 'y_orders']).fill_null(0))
    return candidates


def add_featues(candidates, fold, config):

    print(0, candidates.shape)
    candidates = candidates.groupby(['session', 'aid']).min()
    print(0, candidates.shape)

    # candidates.write_parquet(f'{fold}candidates_raw.parquet')

    feats_obj = LastInteraction(
        fold, 'session_feats', config.data_path)
    feats = feats_obj.load_feature_file()
    feats = candidates.join(feats, on=['session', 'aid'], how='left')
    feats = feats_obj.fill_null(feats)
    print(1, feats.shape)
    del candidates

    if COMPUTE_BPR:
        feats = feats.select(pl.col(['session', 'aid']))

    bprf = BPRFeatures(fold=fold, name='bpr_score', data_path='../data/', cart_weight=2, order_weight=3, agg_func='max', iterations=100,
                       days_of_train=7, learning_rate=.25, regularization=.0004, use_okapi=True, okapi_K1=2.7, okapi_B=.06, factors=256)

    batch = 0
    batch_size = 25_000_000_000
    while batch * batch_size < feats.shape[0]:
        print('BPR score', batch)
        feats_batch = feats[batch * batch_size: (batch + 1) * batch_size]
        bpr_feat = bprf.prepare_features(feats_batch)
        batch += 1
        gc.collect()
    del feats_batch

    gc.collect()
    feats = feats.join(bpr_feat, how='left', on=['session', 'aid'])
    feats = bprf.fill_null(feats)
    del bpr_feat, bprf

    session_click_feats_obj = LastInteraction(
        fold, 'session_clicks_feats', config.data_path, 0)
    session_click_feats = session_click_feats_obj.load_feature_file()

    feats = feats.join(session_click_feats, how='left', on=['session', 'aid'])
    feats = session_click_feats_obj.fill_null(feats)
    del session_click_feats, session_click_feats_obj
    print(2, feats.shape)

    session_carts_feats_obj = LastInteraction(
        fold, 'session_carts_feats', config.data_path, 1)
    session_carts_feats = session_carts_feats_obj.load_feature_file()

    feats = feats.join(session_carts_feats, how='left', on=['session', 'aid'])
    feats = session_carts_feats_obj.fill_null(feats)
    del session_carts_feats, session_carts_feats_obj
    print(3, feats.shape)

    session_orders_feats_obj = LastInteraction(
        fold, 'session_orders_feats', config.data_path, 2)
    session_orders_feats = session_orders_feats_obj.load_feature_file()

    feats = feats.join(session_orders_feats, how='left', on=['session', 'aid'])
    feats = session_orders_feats_obj.fill_null(feats)
    del session_orders_feats, session_orders_feats_obj
    print(4, feats.shape)

    feats = feats.with_column(((pl.col('cart_interaction_last_time') >= 0) & (
        pl.col('order_interaction_last_time') < 0)).alias('cart_without_order1'))
    feats = feats.with_column(((pl.col('cart_interaction_last_time') >= 0) & (pl.col('order_interaction_last_time') >= 0) & (
        pl.col('order_interaction_last_time') > pl.col('cart_interaction_last_time'))).alias('cart_without_order2'))
    feats = feats.with_column((pl.col('cart_without_order1') | pl.col('cart_without_order2')).alias(
        'cart_without_order')).drop(['cart_without_order1', 'cart_without_order2'])

    feats = feats.with_column(((pl.col('click_interaction_last_time') >= 0) & (
        pl.col('order_interaction_last_time') < 0)).alias('click_without_order1'))
    feats = feats.with_column(((pl.col('click_interaction_last_time') >= 0) & (pl.col('order_interaction_last_time') >= 0) & (
        pl.col('order_interaction_last_time') > pl.col('click_interaction_last_time'))).alias('click_without_order2'))
    feats = feats.with_column((pl.col('click_without_order1') | pl.col('click_without_order2')).alias(
        'click_without_order')).drop(['click_without_order1', 'click_without_order2'])

    feats = feats.with_column(((pl.col('click_interaction_last_time') >= 0) & (
        pl.col('cart_interaction_last_time') < 0)).alias('click_without_cart1'))
    feats = feats.with_column(((pl.col('click_interaction_last_time') >= 0) & (pl.col('cart_interaction_last_time') >= 0) & (
        pl.col('cart_interaction_last_time') > pl.col('click_interaction_last_time'))).alias('click_without_cart2'))
    feats = feats.with_column((pl.col('click_without_cart1') | pl.col('click_without_cart2')).alias(
        'click_without_cart')).drop(['click_without_cart1', 'click_without_cart2'])

    print(5, feats.shape)
    aid_feats_obj = AidFeatures(
        data_path=config.data_path, fold=fold, name='aid_stats')
    aid_feats = aid_feats_obj.load_feature_file()

    feats = feats.join(aid_feats, how='left', on='aid')
    feats = aid_feats_obj.fill_null(feats)
    del aid_feats, aid_feats_obj
    print(6, feats.shape)

    session_feats_obj = SessionFeatures(
        data_path=config.data_path, fold=fold, name='session_stats')
    session_feats = session_feats_obj.load_feature_file()

    feats = feats.join(session_feats, how='left', on='session')
    feats = session_feats_obj.fill_null(feats)
    del session_feats, session_feats_obj
    print(7, feats.shape)

    session_clicks_feats_obj = SessionFeatures(
        data_path=config.data_path, fold=fold, name='session_clicks_stats', event_type=0)
    session_clicks_feats = session_clicks_feats_obj.load_feature_file()

    feats = feats.join(session_clicks_feats, how='left', on='session')
    feats = session_clicks_feats_obj.fill_null(feats)
    del session_clicks_feats, session_clicks_feats_obj
    print(8, feats.shape)

    session_carts_feats_obj = SessionFeatures(
        data_path=config.data_path, fold=fold, name='session_carts_stats', event_type=1)
    session_carts_feats = session_carts_feats_obj.load_feature_file()

    feats = feats.join(session_carts_feats, how='left', on='session')
    feats = session_carts_feats_obj.fill_null(feats)
    del session_carts_feats, session_carts_feats_obj
    print(9, feats.shape)

    session_orders_feats_obj = SessionFeatures(
        data_path=config.data_path, fold=fold, name='session_orders_stats', event_type=2)
    session_orders_feats = session_orders_feats_obj.load_feature_file()

    feats = feats.join(session_orders_feats, how='left', on='session')
    feats = session_orders_feats_obj.fill_null(feats)
    del session_orders_feats, session_orders_feats_obj
    print(10, feats.shape)

    cvs1_obj = CovisitScore(fold=fold, name='covisit_score1', data_path=config.data_path, type_weight={0: 1, 1: 6, 2: 3},
                            days_back=14, before_time=0, after_time=24 * 60 * 60, left_types=[0, 1, 2], right_types=[0, 1, 2], reco_hist=30)

    cvs1 = cvs1_obj.load_feature_file()

    feats = feats.join(cvs1, how='left', on=['session', 'aid'])
    feats = cvs1_obj.fill_null(feats)
    del cvs1, cvs1_obj
    print(11, feats.shape)

    cvs2_obj = CovisitScore(fold=fold, name='covisit_score2', data_path=config.data_path, type_weight={0: 1, 1: 6, 2: 3},
                            days_back=14, before_time=0, after_time=24 * 60 * 60, left_types=[0, 1, 2], right_types=[0], reco_hist=30)

    cvs2 = cvs2_obj.load_feature_file()

    feats = feats.join(cvs2, how='left', on=['session', 'aid'])
    feats = cvs2_obj.fill_null(feats)
    del cvs2, cvs2_obj
    print(12, feats.shape)

    cvs3_obj = CovisitScore(fold=fold, name='covisit_score3', data_path=config.data_path, type_weight={0: 1, 1: 6, 2: 3},
                            days_back=14, before_time=0, after_time=24 * 60 * 60, left_types=[0, 1, 2], right_types=[1], reco_hist=30)

    cvs3 = cvs3_obj.load_feature_file()

    feats = feats.join(cvs3, how='left', on=['session', 'aid'])
    feats = cvs3_obj.fill_null(feats)
    del cvs3, cvs3_obj
    print(13, feats.shape)

    cvs4_obj = CovisitScore(fold=fold, name='covisit_score4', data_path=config.data_path, type_weight={0: 1, 1: 6, 2: 3},
                            days_back=14, before_time=0, after_time=24 * 60 * 60, left_types=[0, 1, 2], right_types=[2], reco_hist=30)

    cvs4 = cvs4_obj.load_feature_file()

    feats = feats.join(cvs4, how='left', on=['session', 'aid'])
    feats = cvs4_obj.fill_null(feats)
    del cvs4, cvs4_obj
    print(14, feats.shape)

    cvs5_obj = CovisitScore(fold=fold, name='covisit_score5', data_path=config.data_path, type_weight={0: 1, 1: 6, 2: 3},
                            days_back=14, before_time=0, after_time=24 * 60 * 60, left_types=[1, 2], right_types=[1, 2], reco_hist=30)

    cvs5 = cvs5_obj.load_feature_file()

    feats = feats.join(cvs5, how='left', on=['session', 'aid'])
    feats = cvs5_obj.fill_null(feats)
    del cvs5, cvs5_obj
    print(15, feats.shape)

    feats = feats.with_column((pl.col(
        'aid_sess_cnt') / pl.col('session_session_aid_cnt_m')).alias('session_aid_pop_rel'))
    feats = feats.with_column((pl.col('aid_sess_click_cnt') /
                              pl.col('click_session_aid_cnt_m')).alias('click_aid_pop_rel'))
    feats = feats.with_column((pl.col(
        'aid_sess_cart_cnt') / pl.col('cart_session_aid_cnt_m')).alias('cart_aid_pop_rel'))
    feats = feats.with_column((pl.col('aid_sess_order_cnt') /
                              pl.col('order_session_aid_cnt_m')).alias('order_aid_pop_rel'))

    print(16, feats.shape)

    
    
    
    
    print('FEATURES:\n', feats.columns)
    return feats


class Feature():
    def __init__(self, fold, name, data_path):
        self.fold = fold
        self.name = name
        self.data_path = data_path

    def prepare_features(self):
        raise NotImplementedError

    def fill_null(self):
        raise NotImplementedError

    def load_feature_file(self):
        feature_file = Path(
            f'{self.data_path}features/{self.fold}{self.name}.parquet')
        if not feature_file.is_file():
            print(f'preparing features {self.fold}{self.name}')
            self.prepare_features()
        return pl.read_parquet(feature_file.as_posix()).lazy()


class LastInteraction(Feature):
    def __init__(self, fold, name, data_path, event_type=None):
        super().__init__(fold=fold, name=name, data_path=data_path)
        self.fold = fold
        self.event_type = event_type
        if event_type is not None:
            self.event_type_dict = {0: 'click', 1: 'cart', 2: 'order'}
            self.event_type_str = self.event_type_dict[event_type]
        else:
            self.event_type_str = 'session'

    def prepare_features(self):
        df = pl.read_parquet(
            f'{self.data_path}raw/{self.fold}test.parquet').lazy()
        session_ts_max = df.groupby('session').agg(
            pl.col('ts').max().alias('ts_max'))
        if self.event_type:
            df = df.filter(pl.col('type') == self.event_type)
        df = df.join(session_ts_max, on='session')
        df = df.with_column((pl.col('ts_max') - pl.col('ts')
                             ).alias('ts_since_interaction'))

        df = df.groupby(['session', 'aid']).agg([
            pl.col('ts').count().alias(
                f'{self.event_type_str}_interaction_cnt'),
            pl.col('ts_since_interaction').min().alias(
                f'{self.event_type_str}_interaction_last_time'),
            pl.col('ts').max(),
            
            
        ])

        df = df.with_columns(session_ts_max- data['ts'].alias('ts_diff'))
        df = df.with_columns(df.groupby(['session', 'aid']).agg([
            pl.col('ts_diff').min()+1,
            pl.col('type').last()
                ])  

        
        df = df.with_columns(pl.col(['session', 'aid', f'{self.event_type_str}_interaction_cnt',
                                     f'{self.event_type_str}_interaction_last_time']))
        
        
        df = df.with_columns([
            ((pl.col('ts').cast(pl.Int64) + 7200) *
             1000000).cast(pl.Datetime).dt.weekday().alias(f'{self.event_type_str}_last_weekday'),
            (((pl.col('ts').cast(pl.Int64) + 7200) * 1000000).cast(pl.Datetime).dt.hour() /
             6).cast(pl.Int16).alias(f'{self.event_type_str}_last_time_of_day')
        ]).drop('ts')

        df = df.collect()

        df.write_parquet(
            f'{self.data_path}features/{self.fold}{self.name}.parquet')

    def fill_null(self, df):
        df = df.with_column(
            pl.col(f'{self.event_type_str}_interaction_cnt').fill_null(0))
        df = df.with_column(
            pl.col(f'{self.event_type_str}_interaction_last_time').fill_null(-1))
        df = df.with_column(
            pl.col(f'{self.event_type_str}_last_weekday').fill_null(99))
        df = df.with_column(
            pl.col(f'{self.event_type_str}_last_time_of_day').fill_null(99))

        return df


class AidFeatures(Feature):
    def __init__(self, fold, name, data_path):
        super().__init__(fold=fold, name=name, data_path=data_path)
        self.fold = fold

    def prepare_features(self):
        df = pl.read_parquet(
            f'{self.data_path}raw/{self.fold}test.parquet')
        df2 = pl.read_parquet(
            f'{self.data_path}raw/{self.fold}train.parquet')
        max_train_ts = df2.select(pl.col('ts').max().alias('max'))[0, 0]
        df2 = df2.filter(pl.col('ts') > (max_train_ts - (3600 * 24 * 7)))
        df = pl.concat([df, df2])
        del df2, max_train_ts
        ts_max = df.select(pl.col('ts').max())[0, 0]
        ts_min = df.select(pl.col('ts').min())[0, 0]

        aid_cnt_total = df.select('session').shape[0]
        session_cnt_total = df.select('session').n_unique()

        df = df.with_columns([
            ((pl.col('ts').cast(pl.Int64)) *
             1000000).cast(pl.Datetime)]).dt.hour.alias('hour')
        
            
        aid_stats = df.groupby('aid').agg([
            pl.col('ts').max().alias('aid_max_ts'),
            pl.col('ts').min().alias('aid_min_ts'),
            pl.col('session').count().alias('aid_cnt'),
            pl.col('session').n_unique().alias('aid_sess_cnt'),
            pl.col('hour').std().alias('aid_std_hour'),
            pl.col('hour').max().alias('aid_max_hour'),
            pl.col('hour').min().alias('aid_min_hour'),       
            pl.col('hour').mean().alias('aid_mean_hour'),     
        ])
        aid_stats = aid_stats.with_column(
            (ts_max - pl.col('aid_max_ts')).alias('aid_max_ts'))
        aid_stats = aid_stats.with_column(pl.col('aid_min_ts') - ts_min)

        aid_stats = aid_stats.with_column(
            pl.col('aid_cnt') / aid_cnt_total * 100_000)
        aid_stats = aid_stats.with_column(
            pl.col('aid_sess_cnt') / session_cnt_total * 100_000)

        aid_click_cnt_total = df.filter(
            pl.col('type') == 0).select('session').shape[0]
        session_click_cnt_total = df.filter(
            pl.col('type') == 0).select('session').n_unique()

  
        
        
        aid_click_stats = df.filter(pl.col('type') == 0).groupby('aid').agg([
            pl.col('ts').max().alias('aid_click_max_ts'),
            pl.col('ts').min().alias('aid_click_min_ts'),
            pl.col('session').count().alias('aid_click_cnt'),
            pl.col('session').n_unique().alias('aid_sess_click_cnt'),

            pl.col('hour').std().alias('aid_click_std_hour'),
            pl.col('hour').max().alias('aid_click_max_hour'),
            pl.col('hour').min().alias('aid_click_min_hour'),       
            pl.col('hour').mean().alias('aid_click_mean_hour'),   
        
        ])
        aid_click_stats = aid_click_stats.with_column(
            (pl.col('aid_click_max_ts') - ts_max).abs())
        aid_click_stats = aid_click_stats.with_column(
            pl.col('aid_click_min_ts') - ts_min)

        aid_click_stats = aid_click_stats.with_column(
            pl.col('aid_click_cnt').alias('aid_click_cnt_raw'))
        aid_click_stats = aid_click_stats.with_column(
            pl.col('aid_click_cnt') / aid_click_cnt_total * 100_000)
        
        
        aid_click_stats = aid_click_stats.with_column(
            pl.col('aid_sess_click_cnt') / session_click_cnt_total * 100_000)

        aid_cart_cnt_total = df.filter(
            pl.col('type') == 1).select('session').shape[0]
        session_cart_cnt_total = df.filter(
            pl.col('type') == 1).select('session').n_unique()

        aid_cart_stats = df.filter(pl.col('type') == 1).groupby('aid').agg([
            pl.col('ts').max().alias('aid_cart_max_ts'),
            pl.col('ts').min().alias('aid_cart_min_ts'),
            pl.col('session').count().alias('aid_cart_cnt'),
            pl.col('session').n_unique().alias('aid_sess_cart_cnt'),
            pl.col('hour').std().alias('aid_cart_std_hour'),
            pl.col('hour').max().alias('aid_cart_max_hour'),
            pl.col('hour').min().alias('aid_cart_min_hour'),       
            pl.col('hour').mean().alias('aid_cart_mean_hour'), 
        ])
        aid_cart_stats = aid_cart_stats.with_column(
            (pl.col('aid_cart_max_ts') - ts_max).abs())
        aid_cart_stats = aid_cart_stats.with_column(
            pl.col('aid_cart_min_ts') - ts_min)

        aid_cart_stats = aid_cart_stats.with_column(
            pl.col('aid_cart_cnt').alias('aid_cart_cnt_raw'))
        aid_cart_stats = aid_cart_stats.with_column(
            (pl.col('aid_cart_cnt') / aid_cart_cnt_total * 100_000))
        aid_cart_stats = aid_cart_stats.with_column(
            pl.col('aid_sess_cart_cnt') / session_cart_cnt_total * 100_000)

        aid_cart_stats = aid_cart_stats.with_column(
            pl.col('aid_cart_cnt').alias('aid_cart_cnt_raw'))
        aid_order_cnt_total = df.filter(
            pl.col('type') == 2).select('session').shape[0]
        session_order_cnt_total = df.filter(
            pl.col('type') == 2).select('session').n_unique()

        aid_order_stats = df.filter(pl.col('type') == 2).groupby('aid').agg([
            pl.col('ts').max().alias('aid_order_max_ts'),
            pl.col('ts').min().alias('aid_order_min_ts'),
            pl.col('session').count().alias('aid_order_cnt'),
            pl.col('session').n_unique().alias('aid_sess_order_cnt'),

            pl.col('hour').std().alias('aid_order_std_hour'),
            pl.col('hour').max().alias('aid_order_max_hour'),
            pl.col('hour').min().alias('aid_order_min_hour'),       
            pl.col('hour').mean().alias('aid_order_mean_hour'), 
        
        ])
        aid_order_stats = aid_order_stats.with_column(
            (pl.col('aid_order_max_ts') - ts_max).abs())
        aid_order_stats = aid_order_stats.with_column(
            pl.col('aid_order_min_ts') - ts_min)

        aid_order_stats = aid_order_stats.with_column(
            pl.col('aid_order_cnt').alias('aid_order_cnt_raw'))
        aid_order_stats = aid_order_stats.with_column(
            pl.col('aid_order_cnt') / aid_order_cnt_total * 100_000)
        aid_order_stats = aid_order_stats.with_column(
            pl.col('aid_sess_order_cnt') / session_order_cnt_total * 100_000)
    
        aid_stats = aid_stats.join(aid_click_stats, on='aid', how='left')
        aid_stats = aid_stats.join(aid_cart_stats, on='aid', how='left')
        aid_stats = aid_stats.join(aid_order_stats, on='aid', how='left')

        aid_stats = aid_stats.with_columns(
            pl.col(['aid_max_ts', 'aid_min_ts',
                    'aid_click_max_ts', 'aid_click_min_ts', 
                    'aid_cart_max_ts', 'aid_cart_min_ts', 
                    'aid_order_max_ts', 'aid_order_min_ts',
                    'aid_max_hour', 'aid_min_hour', 
                    'aid_click_max_hour', 'aid_click_min_hour', 'aid_click_std_hour', 
                    'aid_cart_max_hour', 'aid_cart_min_hour',  'aid_cart_std_hour', 
                    'aid_order_max_hour', 'aid_order_min_hour', 'aid_order_std_hour',
                    'aid_mean_hour', 'aid_click_mean_hour', 'aid_click_std_hour',
                    'aid_cart_mean_hour', 'aid_order_mean_hour', 'aid_order_std_hour'
                   ]).fill_null(-999999))
        aid_stats = aid_stats.with_columns(
            pl.col(['aid_cnt', 'aid_click_cnt', 'aid_cart_cnt', 'aid_order_cnt', 'aid_sess_cnt', 'aid_sess_click_cnt', 'aid_sess_cart_cnt', 'aid_sess_order_cnt']).fill_null(0))

        aid_stats = aid_stats.with_column(
            (pl.col('aid_cart_cnt_raw') / pl.col('aid_click_cnt_raw') * 100).alias('click_to_cart'))
        aid_stats = aid_stats.with_column(
            (pl.col('aid_order_cnt_raw') / pl.col('aid_click_cnt_raw') * 100).alias('click_to_order'))
        aid_stats = aid_stats.with_column(
            (pl.col('aid_order_cnt_raw') / pl.col('aid_cart_cnt_raw') * 100).alias('cart_to_order'))

        aid_stats = aid_stats.drop(
            ['aid_click_cnt_raw', 'aid_cart_cnt_raw', 'aid_order_cnt_raw'])

        aid_stats.write_parquet(
            f'{self.data_path}features/{self.fold}{self.name}.parquet')

    def fill_null(self, df):
        df = df.with_columns(
            pl.col(['aid_max_ts', 'aid_min_ts', 'aid_click_max_ts', 'aid_click_min_ts', 'aid_cart_max_ts', 'aid_cart_min_ts', 'aid_order_max_ts', 'aid_order_min_ts','aid_max_hour', 'aid_min_hour', 'aid_click_max_hour', 'aid_click_min_hour', 'aid_cart_max_hour', 'aid_cart_min_hour', 'aid_order_max_hour', 'aid_order_min_hour','aid_mean_hour', 'aid_click_mean_hour','aid_cart_mean_hour', 'aid_order_mean_hour']).fill_null(-999999))
        df = df.with_columns(
            pl.col(['aid_cnt', 'aid_click_cnt', 'aid_cart_cnt', 'aid_order_cnt', 'aid_sess_cnt', 'aid_sess_click_cnt', 'aid_sess_cart_cnt', 'aid_sess_order_cnt']).fill_null(0))
        df = df.with_columns(
            pl.col(['click_to_cart', 'click_to_order', 'cart_to_order']).fill_null(-1))
        return df


class SessionFeatures(Feature):
    def __init__(self, fold, name, data_path, event_type=None):
        super().__init__(fold=fold, name=name, data_path=data_path)
        self.fold = fold
        self.event_type = event_type
        if event_type is not None:
            self.event_type_dict = {0: 'click', 1: 'cart', 2: 'order'}
            self.event_type_str = self.event_type_dict[event_type]
        else:
            self.event_type_str = 'session'

    def prepare_features(self):
        df = pl.read_parquet(
            f'{self.data_path}raw/{self.fold}test.parquet')
        if self.event_type:
            df = df.filter(pl.col('type') == self.event_type)
        aid_cnt = df.groupby('aid').agg(
            pl.col('session').n_unique().alias(f'{self.event_type_str}_aid_cnt'))
        df = df.join(aid_cnt, on='aid')
        df = df.groupby('session').agg([pl.col('aid').count().alias(f'{self.event_type_str}_cnt'),
                                        pl.col('aid').n_unique().alias(
                                            f'{self.event_type_str}_cnt_distinct'),
                                        pl.col(f'{self.event_type_str}_aid_cnt').median().alias(
                                            f'{self.event_type_str}_session_aid_cnt_m'),
                                    
                                        ])
        df = df.with_column((pl.col(f'{self.event_type_str}_cnt') / pl.col(
            f'{self.event_type_str}_cnt_distinct')).alias(f'{self.event_type_str}_events_per_aid'))

        
        df = df.with_columns([
            ((pl.col('ts').cast(pl.Int64)) *
             1000000).cast(pl.Datetime)]).alias('time')
        
        df = df.with_columns(
            pl.col(f'time').dt.hour.alias('hour'),
            pl.col(f'time').dt.hour.alias('day'),
            pl.col(f'time').dt.month.alias('month'),
            pl.col(f'time').dt.month.alias('year'),
                            )
        df = df.groupby(['session','hour','day','month']).agg(
            pl.col(f'ts').dt.count.alias('hour_count'),
            pl.col(f'ts').dt.max.alias('hour_max'),
            pl.col(f'ts').dt.min.alias('hour_min'),
            pl.col(f'ts').dt.min.alias('hour_std'),
            pl.col('session').count().alias('aid_click_cnt'),
            pl.col('session').n_unique().alias('aid_sess_click_cnt'),
            
            
        )
        
        
        # Average gap per hour
        df = df.with_columns([(
            (pl.col(f'hour_max')-pl.col(f'hour_min'))/pl.col(f'hour_count')
        ).alias(f'{self.event_type_str}_session_hour_avg_gap'),
            ((pl.col(f'hour_max')-pl.col(f'hour_min'))/pl.col(f'hour_count')
            ).log().alias(f'{self.event_type_str}_session_hour_avg_gap_log'),
            (pl.col(f'{self.event_type_str}_cnt')/pl.col(f'{self.event_type_str}_aid_cnt')
            ).alias(f'{self.event_type_str}_ratio'),
        ])   
            
            
        df.write_parquet(
            f'{self.data_path}features/{self.fold}{self.name}.parquet')

    def fill_null(self, df):
        df = df.with_columns(
            pl.col([f'{self.event_type_str}_cnt', f'{self.event_type_str}_cnt_distinct']).fill_null(0))
        df = df.with_columns(
            pl.col([f'{self.event_type_str}_events_per_aid', f'{self.event_type_str}_session_aid_cnt_m']).fill_null(-1))
        return df


class CovisitScore(Feature):
    def __init__(self, fold, name, data_path, type_weight, days_back, before_time, after_time, left_types, right_types,
                 time_weight_coef=3, normalize=True, session_hist=30, weekdays=None, dayparts=None, reco_hist=30):
        super().__init__(fold=fold, name=name, data_path=data_path)
        self.fold = fold
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

    def prepare_features(self, return_df=False):

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

        df = df.filter(pl.col('n') < self.reco_hist).drop('n')

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
        reco = reco.select(pl.col(['session', 'aid_right', 'wgt']))
        reco.columns = ['session', 'aid', self.name]

        if return_df:
            return reco
        else:
            reco.write_parquet(
                f'{self.data_path}features/{self.fold}{self.name}.parquet')

    def fill_null(self, df):
        df = df.with_columns(
            pl.col([self.name]).fill_null(0))
        return df


class BPRFeatures(Feature):

    def __init__(self, fold, name, data_path, cart_weight, order_weight, agg_func, iterations=100, learning_rate=0.25, regularization=.0005, use_okapi=True, okapi_K1=2.7, okapi_B=.06, days_of_train=7, factors=512, add_train=True):
        super().__init__(fold=fold, name=name, data_path=data_path)
        self.fold = fold
        self.data_path = data_path
        self.add_train = add_train
        self.factors = factors
        self.iterations = iterations
        self.days_of_train = days_of_train
        self.maxs = {'': 1661723999,
                     'valid3__': 1661119199,
                     'valid2__': 1660514399,
                     'valid1__': 1659909599}
        self.max_ts = self.maxs[self.fold]
        self.min_ts = self.max_ts - (24 * 60 * 60 * self.days_of_train)

        self.cart_weight = cart_weight
        self.order_weight = order_weight
        self.mapper = {0: 1, 1: self.cart_weight, 2: self.order_weight}

        self.agg_func = agg_func
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.use_okapi = use_okapi
        self.okapi_K1 = okapi_K1
        self.okapi_B = okapi_B

    def fill_null(self, df):
        df = df.with_columns(
            pl.col(self.name).fill_null(-9))
        return df

    def load_feature_file(self):
        raise (ValueError('use prepare_features method'))

    def prepare_features(self, candidates):
        candidates = candidates.select(['session', 'aid'])
        model_file = Path(
            f'{self.data_path}features/{self.fold}{self.name}_model.pkl')
        if not model_file.is_file():
            print(f'training {self.fold}{self.name}_model')
            self.train_model()

        feature_file = Path(
            f'{self.data_path}features/{self.fold}{self.name}.parquet')
        if feature_file.is_file():
            feat = pl.read_parquet(feature_file.as_posix())
            cands_to_score = candidates.join(
                feat, on=['session', 'aid'], how='left')
            cands_to_score = cands_to_score.filter(pl.col(self.name).is_null())
        else:
            cands_to_score = candidates

        session_inv = pl.read_parquet(
            f'{self.data_path}features/{self.fold}{self.name}_session_inv.parquet')
        aid_inv = pl.read_parquet(
            f'{self.data_path}features/{self.fold}{self.name}_aid_inv.parquet')

        cands_to_score = cands_to_score.join(session_inv, on='session')
        cands_to_score = cands_to_score.join(aid_inv, on='aid')

        if cands_to_score.shape[0] == 0:
            return feat
        else:
            cands_to_score = cands_to_score.drop(['session_idx', 'aid_idx'])

            model = joblib.load(
                f'{self.data_path}features/{self.fold}{self.name}_model.pkl')

            session_inv = session_inv.sort(by='session_idx')
            session_idx = session_inv.select('session').to_numpy().ravel()

            aid_inv = aid_inv.sort(by='aid_idx')
            aid_idx = aid_inv.select('aid').to_numpy().ravel()

            del session_inv, aid_inv

            users_dict = {sess: model.user_factors[i].reshape(
                1, -1) for i, sess in enumerate(session_idx)}
            aids_dict = {aid: model.item_factors[i].reshape(
                1, -1) for i, aid in enumerate(aid_idx)}
            del model

            sessions = []
            aids = []
            user_vec = []
            item_vec = []
            cands_to_score = cands_to_score.to_numpy()
            for i in tqdm(cands_to_score):
                sessions.append(i[0])
                aids.append(i[1])
                user_vec.append(users_dict[i[0]])
                item_vec.append(aids_dict[i[1]])
            print('stacking')
            user_vec = np.vstack(user_vec)
            item_vec = np.vstack(item_vec)
            print('scoring')
            distances = (user_vec * item_vec).sum(1) / (np.sqrt((user_vec *
                                                                 user_vec).sum(1)) * np.sqrt((item_vec * item_vec).sum(1)))
            distances = distances.tolist()

            feat_new = pl.DataFrame([sessions, aids, distances], columns=[
                                    'session', 'aid', self.name])
            del user_vec, item_vec, distances

            feat_new = feat_new.select([pl.col(['session', 'aid']).cast(
                pl.Int32), pl.col(self.name).cast(pl.Float32)])

            if feature_file.is_file():
                feat = pl.concat([feat, feat_new])
                feat.write_parquet(
                    f'{self.data_path}features/{self.fold}{self.name}.parquet')
                del feat_new
                gc.collect()
                return feat
            else:
                feat_new.write_parquet(
                    f'{self.data_path}features/{self.fold}{self.name}.parquet')
                gc.collect()
                return feat_new

    def train_model(self):

        start_time = time.time()
        if self.add_train:
            df1 = pl.read_parquet(
                f'{self.data_path}raw/{self.fold}test.parquet')
            df2 = pl.read_parquet(
                f'{self.data_path}raw/{self.fold}train.parquet')
            df1 = df1.with_column(pl.lit(1, pl.Int8).alias('test'))
            df2 = df2.with_column(pl.lit(0, pl.Int8).alias('test'))
            df = pl.concat([df1, df2], how='vertical')
            del df1, df2
        else:
            df = pl.read_parquet(
                f'{self.data_path}raw/{self.fold}test.parquet')
            df = df.with_column(pl.lit(1, pl.Int8).alias('test'))

        df = df.filter(pl.col('ts') >= self.min_ts)

        df = df.drop('ts')
        df = df.with_column(pl.col('type').apply(lambda x: self.mapper[x]))

        if self.agg_func == 'sum':
            df = df.groupby(['session', 'aid', 'test']).sum()
        elif self.agg_func == 'max':
            df = df.groupby(['session', 'aid', 'test']).max()

        aid_cnt = df.groupby('aid').agg(
            pl.col('session').n_unique().alias('cnt'))
        aid_cnt = aid_cnt.filter(pl.col('cnt') >= 0)
        df = df.join(aid_cnt, on='aid').drop('cnt')
        del aid_cnt
        df = df.with_column(
            (pl.col('session').rank('dense') - 1).alias('session_idx'))
        df = df.with_column((pl.col('aid').rank('dense') - 1).alias('aid_idx'))
        values = df.select(pl.col('type')).to_numpy().ravel()
        session_idx = df.select(pl.col('session_idx')).to_numpy().ravel()
        aid_idx = df.select(pl.col('aid_idx')).to_numpy().ravel()
        aid_session = scipy.sparse.coo_matrix((values, (aid_idx, session_idx)), shape=(np.unique(aid_idx).shape[0],
                                                                                       np.unique(session_idx).shape[0]))
        session_idx = np.unique(session_idx)
        aid_session = aid_session.tocsr()

        start_time = time.time()
        if self.use_okapi:
            aid_session = bm25_weight(
                aid_session, K1=self.okapi_K1, B=self.okapi_B)
        session_aid = aid_session.T.tocsr()
        model = BayesianPersonalizedRanking(factors=self.factors,
                                            learning_rate=self.learning_rate,
                                            regularization=self.regularization,
                                            iterations=self.iterations)

        model.fit(session_aid, show_progress=False)
        joblib.dump(
            model, f'{self.data_path}features/{self.fold}{self.name}_model.pkl')

        session_inv = df.select(pl.col(['session', 'session_idx'])).unique()
        aid_inv = df.select(pl.col(['aid', 'aid_idx'])).unique()

        session_inv = session_inv.select(pl.col('*').cast(pl.Int32))
        aid_inv = aid_inv.select(pl.col('*').cast(pl.Int32))

        session_inv.write_parquet(
            f'{self.data_path}features/{self.fold}{self.name}_session_inv.parquet')
        aid_inv.write_parquet(
            f'{self.data_path}features/{self.fold}{self.name}_aid_inv.parquet')
