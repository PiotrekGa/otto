import polars as pl
from pathlib import Path


def add_labels(candidates, fold, config):
    labels = pl.read_parquet(
        f'{config.data_path}{fold}test_labels.parquet')
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

    aid_feats_obj = AidFeatures(
        data_path=config.data_path, fold=fold, name='aid_stats')
    aid_feats = aid_feats_obj.load_feature_file()

    feats = feats.join(aid_feats, how='left', on='aid')
    feats = aid_feats_obj.fill_null(feats)
    del aid_feats, aid_feats_obj

    session_feats_obj = SessionFeatures(
        data_path=config.data_path, fold=fold, name='session_stats')
    session_feats = session_feats_obj.load_feature_file()

    feats = feats.join(session_feats, how='left', on='session')
    feats = session_feats_obj.fill_null(feats)
    del session_feats, session_feats_obj

    session_clicks_feats_obj = SessionFeatures(
        data_path=config.data_path, fold=fold, name='session_clicks_stats', event_type=0)
    session_clicks_feats = session_clicks_feats_obj.load_feature_file()

    feats = feats.join(session_clicks_feats, how='left', on='session')
    feats = session_clicks_feats_obj.fill_null(feats)
    del session_clicks_feats, session_clicks_feats_obj

    session_carts_feats_obj = SessionFeatures(
        data_path=config.data_path, fold=fold, name='session_carts_stats', event_type=1)
    session_carts_feats = session_carts_feats_obj.load_feature_file()

    feats = feats.join(session_carts_feats, how='left', on='session')
    feats = session_carts_feats_obj.fill_null(feats)
    del session_carts_feats, session_carts_feats_obj

    session_orders_feats_obj = SessionFeatures(
        data_path=config.data_path, fold=fold, name='session_orders_stats', event_type=2)
    session_orders_feats = session_orders_feats_obj.load_feature_file()

    feats = feats.join(session_orders_feats, how='left', on='session')
    feats = session_orders_feats_obj.fill_null(feats)
    del session_orders_feats, session_orders_feats_obj

    feats = feats.with_column((pl.col(
        'aid_sess_cnt') / pl.col('session_session_aid_cnt_m')).alias('session_aid_pop_rel'))
    feats = feats.with_column((pl.col('aid_sess_click_cnt') /
                              pl.col('click_session_aid_cnt_m')).alias('click_aid_pop_rel'))
    feats = feats.with_column((pl.col(
        'aid_sess_cart_cnt') / pl.col('cart_session_aid_cnt_m')).alias('cart_aid_pop_rel'))
    feats = feats.with_column((pl.col('aid_sess_order_cnt') /
                              pl.col('order_session_aid_cnt_m')).alias('order_aid_pop_rel'))

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
            f'{self.data_path}{self.fold}{self.name}.parquet')
        #if not feature_file.is_file():
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
            f'{self.data_path}{self.fold}test.parquet').lazy()
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
            pl.col('ts').max()
        ])

        df = df.with_columns(pl.col(['session', 'aid', f'{self.event_type_str}_interaction_cnt',
                                     f'{self.event_type_str}_interaction_last_time']))
        df = df.with_columns([
            (pl.col('ts').cast(pl.Int64) *
             1000000).cast(pl.Datetime).dt.weekday().alias(f'{self.event_type_str}_last_weekday'),
            ((pl.col('ts').cast(pl.Int64) * 1000000).cast(pl.Datetime).dt.hour() /
             6).cast(pl.Int16).alias(f'{self.event_type_str}_last_time_of_day')
        ]).drop('ts')

        df = df.collect()

        df.write_parquet(
            f'{self.data_path}{self.fold}{self.name}.parquet')

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
            f'{self.data_path}{self.fold}test.parquet')
        df2 = pl.read_parquet(
            f'{self.data_path}{self.fold}train.parquet')
        max_train_ts = df2.select(pl.col('ts').max().alias('max'))[0, 0]
        df2 = df2.filter(pl.col('ts') > (max_train_ts - (3600 * 24 * 7)))
        df = pl.concat([df, df2])
        del df2, max_train_ts
        ts_max = df.select(pl.col('ts').max())[0, 0]
        ts_min = df.select(pl.col('ts').min())[0, 0]
        ts_mean = df.select(pl.col('ts').mean())[0, 0]
        
        aid_stats = df.groupby('aid').agg([
            pl.col('ts').max().alias('aid_max_ts'),
            pl.col('ts').min().alias('aid_min_ts'),
            pl.col('ts').mean().alias('aid_mean_ts'),
            pl.col('ts').last().alias('aid_last_ts'),
            pl.col('ts').first().alias('aid_first_ts'),
            pl.col('ts').quantile(0.25, "nearest").alias('aid_quantile025_ts'),
            pl.col('ts').quantile(0.75, "nearest").alias('aid_quantile075_ts'),
            pl.col('ts').median().alias('aid_median_ts'),
            pl.col('ts').std().alias('aid_std_ts'),
            
            pl.col('ts').drop_nulls().diff().abs().max().suffix("aid_abs_max_diff"),
            pl.col('ts').drop_nulls().diff().abs().min().suffix("aid_abs_min_diff"),
            pl.col('ts').drop_nulls().diff().abs().mean().suffix("aid_abs_mean_diff"),
            pl.col('ts').drop_nulls().diff().abs().median().suffix("aid_abs_median_diff"),
            pl.col('ts').drop_nulls().diff().abs().sum().suffix("aid_abs_sum_diff"),
            pl.col('ts').drop_nulls().diff().pow(2).sum().sqrt().suffix("aid_std_diff"),
            pl.col('ts').drop_nulls().diff(2).pow(2).sum().sqrt().suffix("aid_std_diff2"),
            pl.col('ts').drop_nulls().diff(3).pow(2).sum().sqrt().suffix("aid_std_diff3"),
            
            pl.col('ts').diff().last().suffix("aid_last_diff"),
            pl.col('ts').diff(3).last().suffix("aid_last_diff3"),
            pl.col('ts').diff(4).last().suffix("aid_last_diff4"),
            pl.col('ts').diff(7).last().suffix("aid_last_diff7"),

            
            pl.col('session').count().alias('aid_cnt'),
            pl.col('session').n_unique().alias('aid_sess_cnt'),
            
        ])
        aid_stats = aid_stats.with_column(ts_max - pl.col('aid_max_ts'))
        aid_stats = aid_stats.with_column(pl.col('aid_min_ts') - ts_min)

        
        aid_click_stats = df.filter(pl.col('type') == 0).groupby('aid').agg([
            pl.col('ts').max().alias('aid_click_max_ts'),
            pl.col('ts').min().alias('aid_click_min_ts'),
            pl.col('ts').mean().alias('aid_click_mean_ts'),
            pl.col('ts').last().alias('aid_click_last_ts'),
            pl.col('ts').first().alias('aid_click_first_ts'),
            pl.col('ts').quantile(0.25, "nearest").alias('aid_click_quantile025_ts'),
            pl.col('ts').quantile(0.75, "nearest").alias('aid_click_quantile075_ts'),
            pl.col('ts').median().alias('aid_click_median_ts'),            
            pl.col('ts').std().alias('aid_click_std_ts'),  
            
            pl.col('ts').drop_nulls().diff().abs().max().suffix("aid_click_abs_max_diff"),
            pl.col('ts').drop_nulls().diff().abs().min().suffix("aid_click_abs_min_diff"),
            pl.col('ts').drop_nulls().diff().abs().mean().suffix("aid_click_abs_mean_diff"),
            pl.col('ts').drop_nulls().diff().abs().median().suffix("aid_click_abs_median_diff"),
            pl.col('ts').drop_nulls().diff().abs().sum().suffix("aid_click_abs_sum_diff"),
            pl.col('ts').drop_nulls().diff().pow(2).sum().sqrt().suffix("aid_click_std_diff"),
            pl.col('ts').drop_nulls().diff(2).pow(2).sum().sqrt().suffix("aid_click_std_diff2"),
            pl.col('ts').drop_nulls().diff(3).pow(2).sum().sqrt().suffix("aid_click_std_diff3"),
            
            pl.col('ts').diff().last().suffix("aid_click_last_diff"),
            pl.col('ts').diff(3).last().suffix("aid_click_last_diff3"),
            pl.col('ts').diff(4).last().suffix("aid_click_last_diff4"),
            pl.col('ts').diff(7).last().suffix("aid_click_last_diff7"),
            
            
            pl.col('session').count().alias('aid_click_cnt'),
            pl.col('session').n_unique().alias('aid_sess_click_cnt'),
            
            
            
        ])
        aid_click_stats = aid_click_stats.with_column(
            ts_max - pl.col('aid_click_max_ts'))
        aid_click_stats = aid_click_stats.with_column(
            pl.col('aid_click_min_ts') - ts_min)

        
        aid_cart_stats = df.filter(pl.col('type') == 1).groupby('aid').agg([
            pl.col('ts').max().alias('aid_cart_max_ts'),
            pl.col('ts').min().alias('aid_cart_min_ts'),
            pl.col('ts').mean().alias('aid_cart_mean_ts'),
            pl.col('ts').last().alias('aid_cart_last_ts'),
            pl.col('ts').first().alias('aid_cart_first_ts'),
            pl.col('ts').quantile(0.25, "nearest").alias('aid_cart_quantile025_ts'),
            pl.col('ts').quantile(0.75, "nearest").alias('aid_cart_quantile075_ts'),
            pl.col('ts').median().alias('aid_cart_median_ts'),               
            pl.col('ts').std().alias('aid_cart_std_ts'),    
            pl.col('session').count().alias('aid_cart_cnt'),
            pl.col('session').n_unique().alias('aid_sess_cart_cnt'),
            
            pl.col('ts').drop_nulls().diff().abs().max().suffix("aid_cart_abs_max_diff"),
            pl.col('ts').drop_nulls().diff().abs().min().suffix("aid_cart_abs_min_diff"),
            pl.col('ts').drop_nulls().diff().abs().mean().suffix("aid_cart_abs_mean_diff"),
            pl.col('ts').drop_nulls().diff().abs().median().suffix("aid_cart_abs_median_diff"),
            pl.col('ts').drop_nulls().diff().abs().sum().suffix("aid_cart_abs_sum_diff"),
            pl.col('ts').drop_nulls().diff().pow(2).sum().sqrt().suffix("aid_cart_std_diff"),
            pl.col('ts').drop_nulls().diff(2).pow(2).sum().sqrt().suffix("aid_cart_std_diff2"),
            pl.col('ts').drop_nulls().diff(3).pow(2).sum().sqrt().suffix("aid_cart_std_diff3"),
            
            pl.col('ts').diff().last().suffix("aid_cart_last_diff"),
            pl.col('ts').diff(3).last().suffix("aid_cart_last_diff3"),
            pl.col('ts').diff(4).last().suffix("aid_cart_last_diff4"),
            pl.col('ts').diff(7).last().suffix("aid_cart_last_diff7"),        
        
        
        
        ])
        aid_cart_stats = aid_cart_stats.with_column(
            ts_max - pl.col('aid_cart_max_ts'))
        aid_cart_stats = aid_cart_stats.with_column(
            pl.col('aid_cart_min_ts') - ts_min)

        aid_order_stats = df.filter(pl.col('type') == 2).groupby('aid').agg([
            pl.col('ts').max().alias('aid_order_max_ts'),
            pl.col('ts').min().alias('aid_order_min_ts'),
            pl.col('ts').mean().alias('aid_order_mean_ts'),
            pl.col('ts').last().alias('aid_order_last_ts'),
            pl.col('ts').first().alias('aid_order_first_ts'),
            pl.col('ts').quantile(0.25, "nearest").alias('aid_order_quantile025_ts'),
            pl.col('ts').quantile(0.75, "nearest").alias('aid_order_quantile075_ts'),
            pl.col('ts').median().alias('aid_order_median_ts'),                    
            pl.col('ts').std().alias('aid_order_std_ts'), 
            pl.col('session').count().alias('aid_order_cnt'),
            pl.col('session').n_unique().alias('aid_sess_order_cnt'),
            pl.col('ts').drop_nulls().diff().abs().max().suffix("aid_order_abs_max_diff"),
            pl.col('ts').drop_nulls().diff().abs().min().suffix("aid_order_abs_min_diff"),
            pl.col('ts').drop_nulls().diff().abs().mean().suffix("aid_order_abs_mean_diff"),
            pl.col('ts').drop_nulls().diff().abs().median().suffix("aid_order_abs_median_diff"),
            pl.col('ts').drop_nulls().diff().abs().sum().suffix("aid_order_abs_sum_diff"),
            pl.col('ts').drop_nulls().diff().pow(2).sum().sqrt().suffix("aid_order_std_diff"),
            pl.col('ts').drop_nulls().diff(2).pow(2).sum().sqrt().suffix("aid_order_std_diff2"),
            pl.col('ts').drop_nulls().diff(3).pow(2).sum().sqrt().suffix("aid_order_std_diff3"),
            
            pl.col('ts').diff().last().suffix("aid_order_last_diff"),
            pl.col('ts').diff(3).last().suffix("aid_order_last_diff3"),
            pl.col('ts').diff(4).last().suffix("aid_order_last_diff4"),
            pl.col('ts').diff(7).last().suffix("aid_order_last_diff7"),        
        
        ])
        aid_order_stats = aid_order_stats.with_column(
            ts_max - pl.col('aid_order_max_ts'))
        aid_order_stats = aid_order_stats.with_column(
            pl.col('aid_order_min_ts') - ts_min)

        aid_stats = aid_stats.join(aid_click_stats, on='aid', how='left')
        aid_stats = aid_stats.join(aid_cart_stats, on='aid', how='left')
        aid_stats = aid_stats.join(aid_order_stats, on='aid', how='left')

        aid_stats = aid_stats.with_columns(
            pl.col(['aid_max_ts', 'aid_min_ts', 'aid_mean_ts', 'aid_last_ts', 'aid_first_ts', 'aid_quantile025_ts', 'aid_quantile075_ts', 'aid_median_ts', 'aid_std_ts'
					'aid_click_max_ts', 'aid_click_min_ts', 'aid_click_mean_ts', 'aid_click_last_ts', 'aid_click_first_ts', 'aid_click_quantile025_ts', 'aid_click_quantile075_ts', 'aid_click_median_ts','aid_click_std_ts',
					'aid_cart_max_ts', 'aid_cart_min_ts', 'aid_cart_mean_ts', 'aid_cart_last_ts', 'aid_cart_first_ts', 'aid_cart_quantile025_ts', 'aid_cart_quantile075_ts', 'aid_cart_median_ts','aid_cart_std_ts',
					'aid_order_max_ts', 'aid_order_min_ts', 'aid_order_mean_ts', 'aid_order_last_ts', 'aid_order_first_ts', 'aid_order_quantile025_ts', 'aid_order_quantile075_ts', 'aid_order_median_ts','aid_order_std_ts',
					]).fill_null(999999))
        aid_stats = aid_stats.with_columns(
            pl.col(['aid_cnt', 'aid_click_cnt', 'aid_cart_cnt', 'aid_order_cnt', 'aid_sess_cnt', 'aid_sess_click_cnt', 'aid_sess_cart_cnt', 'aid_sess_order_cnt']).fill_null(0))

        aid_stats = aid_stats.with_column(
            (pl.col('aid_cart_cnt') / pl.col('aid_click_cnt')).alias('click_to_cart'))
        aid_stats = aid_stats.with_column(
            (pl.col('aid_order_cnt') / pl.col('aid_click_cnt')).alias('click_to_order'))
        aid_stats = aid_stats.with_column(
            (pl.col('aid_order_cnt') / pl.col('aid_cart_cnt')).alias('cart_to_order'))

        aid_stats.write_parquet(
            f'{self.data_path}{self.fold}{self.name}.parquet')

    def fill_null(self, df):
        df = df.with_columns(
            pl.col(['aid_max_ts', 'aid_min_ts', 'aid_click_max_ts', 'aid_click_min_ts', 'aid_cart_max_ts', 'aid_cart_min_ts', 'aid_order_max_ts', 'aid_order_min_ts']).fill_null(999999))
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
            f'{self.data_path}{self.fold}test.parquet')
        if self.event_type:
            df = df.filter(pl.col('type') == self.event_type)
        aid_cnt = df.groupby('aid').agg(
            pl.col('session').n_unique().alias(f'{self.event_type_str}_aid_cnt'))
        df = df.join(aid_cnt, on='aid')
        df = df.groupby('session').agg([pl.col('aid').count().alias(f'{self.event_type_str}_cnt'),
                                        pl.col('aid').n_unique().alias(
                                            f'{self.event_type_str}_cnt_distinct'),
                                        pl.col(f'{self.event_type_str}_aid_cnt').median().alias(
                                            f'{self.event_type_str}_session_aid_cnt_m')
                                        ])
        df = df.with_column((pl.col(f'{self.event_type_str}_cnt') / pl.col(
            f'{self.event_type_str}_cnt_distinct')).alias(f'{self.event_type_str}_events_per_aid'))

        df.write_parquet(
            f'{self.data_path}{self.fold}{self.name}.parquet')

    def fill_null(self, df):
        df = df.with_columns(
            pl.col([f'{self.event_type_str}_cnt', f'{self.event_type_str}_cnt_distinct']).fill_null(0))
        df = df.with_columns(
            pl.col([f'{self.event_type_str}_events_per_aid', f'{self.event_type_str}_session_aid_cnt_m']).fill_null(-1))
        return df
