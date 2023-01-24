import polars as pl
import lightgbm as lgb
from catboost import CatBoostRanker, Pool
from copy import deepcopy
import numpy as np
import time


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

    df = candidates.select(pl.col(['session', 'aid', train_column]))
    df = df.sample(frac=1., shuffle=True, seed=42)
    df = df.with_column(pl.col(train_column).cast(pl.Int32))
    pos_cnt = df.groupby('session').agg(
        pl.col(train_column).sum().alias('neg_cnt'))
    pos_cnt = pos_cnt.with_column(
        pl.col('neg_cnt') * config.max_negative_candidates)
    df = df.join(pos_cnt, on='session')
    del pos_cnt

    df = df.with_column(
        pl.col('session').cumcount().over(['session', train_column]).alias('rank'))
    df = df.filter((pl.col(train_column) == 1) | (
        pl.col('rank') <= pl.col('neg_cnt'))).drop(['rank', 'neg_cnt', train_column])

    candidates = candidates.join(df, on=['session', 'aid'])
    del df

    return candidates


def train_rerank_model(candidates, train_column, config):

    candidates = candidates.sort(by='session')
    queries_train = candidates.select('session').to_numpy().ravel()

    train_baskets = candidates.groupby(['session']).agg(
        pl.col('aid').count().alias('basket'))
    train_baskets = train_baskets.select(pl.col('basket'))
    train_baskets = train_baskets.to_numpy().ravel()

    y = candidates.select(pl.col(train_column)).to_numpy().ravel()
    candidates = candidates.select(
        pl.col(config.features)).to_numpy()

    print(f'training lgbm model {train_column}')
    train_dataset = lgb.Dataset(
        data=candidates, label=y, group=train_baskets)
    start_time = time.time()
    model1 = lgb.train(train_set=train_dataset,
                       params=config.model_param)
    print("model lgbm train time --- %s seconds ---" %
          (time.time() - start_time))


#     print(f'training catboost model {train_column}')
#     train_pool = Pool(
#         data=candidates,
#         label=y,
#         group_id=queries_train
#     )
#     del candidates, y, queries_train, train_baskets


#     start_time = time.time()

#     default_parameters = {
#     'iterations': 1000,
#     'verbose': False,
#     'random_seed': 0,
#     }

#     parameters = {}

#     def fit_model(loss_function, additional_params=None, train_pool=train_pool):
#         parameters = deepcopy(default_parameters)
#         parameters['loss_function'] = loss_function
#         parameters['train_dir'] = loss_function

#         if additional_params is not None:
#             parameters.update(additional_params)

#         model = CatBoostRanker(**parameters)
#         model.fit(train_pool, plot=False, silent=True)

#         return model

#     model2 = fit_model('YetiRankPairwise', {'custom_metric': ['RecallAt:top=10']})

#     print("model catboost train time --- %s seconds ---" % (time.time() - start_time))

    return [model1]


def select_recommendations(candidates, event_type_str, models, config, k=20):
    print(f'scoring candidates {event_type_str}')

    print(candidates.shape)
    print(candidates.select(pl.col(['session', 'aid'])).unique().shape)

    batch_size = 1-00_000
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
        scores = np.vstack([model.predict(x) for model in models]).mean(0)
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
