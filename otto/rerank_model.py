import polars as pl
import lightgbm as lgb
import numpy as np


def sample_candidates(candidates, train_column, config):
    print(f'preparing training data {train_column}')

    if config.sample_size is not None:
        candidates = candidates[:config.sample_size]
    cols_to_float32 = []
    for i in zip(candidates.columns, candidates.dtypes):
        if i[1].__name__ == 'Float64':
            cols_to_float32.append(i[0])

    candidates = candidates.with_columns(
        pl.col(cols_to_float32).cast(pl.Float32))
    candidates = candidates.sort(by='session')
    non_neg = candidates.groupby('session').agg(
        [pl.col(train_column).max().alias('is_positive'), pl.col(train_column).min().alias('is_negative')])
    non_neg = non_neg.filter(pl.col('is_positive') > 0).filter(
        pl.col('is_negative') == 0).select(pl.col('session'))
    candidates = candidates.join(non_neg, on='session', how='inner')
    del non_neg
    candidates = candidates.sample(
        frac=1., shuffle=True, seed=42)

    candidates = candidates.with_column(
        pl.col('session').cumcount().over(['session', train_column]).alias('rank'))
    candidates = candidates.filter((pl.col(train_column) == 1) | (
        pl.col('rank') <= config.max_negative_candidates)).drop('rank')
    return candidates


def train_rerank_model(candidates, train_column, config):

    candidates = candidates.sort(by='session')
    train_baskets = candidates.groupby(['session']).agg(
        pl.col('aid').count().alias('basket'))
    train_baskets = train_baskets.select(pl.col('basket'))
    train_baskets = train_baskets.to_numpy().ravel()

    y = candidates.select(pl.col(train_column)).to_numpy().ravel()
    candidates = candidates.select(
        pl.col(config.features)).to_numpy()

    print(f'training model {train_column}')
    train_dataset = lgb.Dataset(
        data=candidates, label=y, group=train_baskets)
    del candidates, y, train_baskets
    model = lgb.train(train_set=train_dataset,
                      params=config.model_param)
    return model


def select_recommendations(candidates, event_type_str, model, config, k=20):
    print(f'scoring candidates {event_type_str}')

    batch_size = 100_000
    batch_num = 0
    sessions = candidates.select(pl.col('session').unique())
    sessions_cnt = sessions.shape[0]

    recommendations = []
    while batch_num * batch_size < sessions_cnt:
        print('SCORING BATCH', batch_num)
        batch_sessions = sessions[batch_num *
                                  batch_size: (batch_num + 1) * batch_size]
        batch_candidates = candidates.join(batch_sessions, on='session')

        x = batch_candidates.select(
            pl.col(config.features)).to_numpy().astype(np.float32)
        scores = model.predict(x)
        batch_candidates_scored = batch_candidates.select(
            pl.col(['session', 'aid']))
        batch_candidates_scored = batch_candidates_scored.with_column(
            pl.lit(scores).alias('score'))

        batch_candidates_scored = batch_candidates_scored.sort(
            by=['session', 'score'], reverse=[False, True])

        batch_recommendations = batch_candidates_scored.groupby(
            'session').agg(pl.col('aid'))

        batch_recommendations = batch_recommendations.select([(pl.col('session').cast(str) + pl.lit(f'_{event_type_str}')).alias('session_type'), pl.col(
            'aid').apply(lambda x: ' '.join([str(i) for i in x[:k]])).alias('labels')])

        recommendations.append(batch_recommendations)
        batch_num += 1

    recommendations = pl.concat(recommendations)
    return recommendations


def select_perfect_recommendations(candidates, event_type_str, k=20):
    print(f'selecting perfect candidates {event_type_str}')
    col_name = 'y_' + event_type_str
    candidates_scored = candidates.select([
        pl.col('session'), pl.col('aid'), pl.col(col_name).alias('score')])
    candidates_scored = candidates_scored.sort(
        by=['session', 'score'], reverse=[False, True])

    recommendations = candidates_scored.groupby(
        'session').agg(pl.col('aid'))

    recommendations = recommendations.select([(pl.col('session').cast(str) + pl.lit(f'_{event_type_str}')).alias('session_type'), pl.col(
        'aid').apply(lambda x: ' '.join([str(i) for i in x[:k]])).alias('labels')])
    return recommendations
