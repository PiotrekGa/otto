import polars as pl
from pathlib import Path


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

    feats_obj = LastInteraction(
        fold, 'session_feats', config.data_path)
    feats = feats_obj.load_feature_file()
    feats = candidates.join(feats, on=['session', 'aid'], how='left')
    feats = feats_obj.fill_null(feats)

    session_click_feats_obj = LastInteraction(
        fold, 'session_clicks_feats', config.data_path, 0)
    session_click_feats = session_click_feats_obj.load_feature_file()

    feats = feats.join(session_click_feats, how='left', on=['session', 'aid'])
    feats = session_click_feats_obj.fill_null(feats)
    del session_click_feats, session_click_feats_obj

    session_carts_feats_obj = LastInteraction(
        fold, 'session_carts_feats', config.data_path, 1)
    session_carts_feats = session_carts_feats_obj.load_feature_file()

    feats = feats.join(session_carts_feats, how='left', on=['session', 'aid'])
    feats = session_carts_feats_obj.fill_null(feats)
    del session_carts_feats, session_carts_feats_obj

    session_orders_feats_obj = LastInteraction(
        fold, 'session_orders_feats', config.data_path, 2)
    session_orders_feats = session_orders_feats_obj.load_feature_file()

    feats = feats.join(session_orders_feats, how='left', on=['session', 'aid'])
    feats = session_orders_feats_obj.fill_null(feats)
    del session_orders_feats, session_orders_feats_obj

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
        df = df.groupby(['session', 'aid']).agg([pl.col('ts').count().alias(
            f'{self.event_type_str}_interaction_cnt'), pl.col('ts_since_interaction').min().alias(f'{self.event_type_str}_interaction_last_time')])
        df = df.with_columns(pl.col(['session', 'aid', f'{self.event_type_str}_interaction_cnt',
                                     f'{self.event_type_str}_interaction_last_time']))
        df = df.collect()
        df.write_parquet(
            f'{self.data_path}features/{self.fold}{self.name}.parquet')

    def fill_null(self, df):
        df = df.with_column(
            pl.col(f'{self.event_type_str}_interaction_cnt').fill_null(0))
        df = df.with_column(
            pl.col(f'{self.event_type_str}_interaction_last_time').fill_null(-1))
        return df
