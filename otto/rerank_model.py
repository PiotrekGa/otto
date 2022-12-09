import polars as pl
import lightgbm as lgb


def train_rerank_model(candidates, train_column, config):

    candidates = candidates.sort(by='session')

    non_neg = candidates.groupby('session').agg(
        [pl.col(train_column).max().alias('is_positive'), pl.col(train_column).min().alias('is_negative')])
    non_neg = non_neg.filter(pl.col('is_positive') > 0).filter(
        pl.col('is_negative') == 0).select(pl.col('session'))
    candidates = candidates.join(non_neg, on='session', how='inner')
    del non_neg

    candidates = candidates.sample(frac=1., shuffle=True, seed=42)
    positive_cands = candidates.filter(pl.col(train_column) == 1)
    negative_cands = candidates.filter(pl.col(train_column) == 0)

    negative_cands = negative_cands.sort(by='session')
    negative_cands = negative_cands.with_column(
        pl.col('session').count().over('session').alias('rank'))
    negative_cands = negative_cands.filter(
        pl.col('rank') <= config.max_negative_candidates).drop('rank')

    candidates = pl.concat([positive_cands, negative_cands]).sample(
        frac=1., shuffle=True, seed=42)
    del positive_cands, negative_cands

    candidates = candidates.sort(by='session')

    y = candidates.select(pl.col(train_column)).to_numpy().ravel()
    x = candidates.select(pl.col(config.features)).to_numpy()
    train_baskets = candidates.groupby(['session']).agg(
        pl.col('aid').count().alias('basket'))
    train_baskets = train_baskets.select(pl.col('basket'))
    train_baskets = train_baskets.to_numpy().ravel()

    del candidates

    train_dataset = lgb.Dataset(
        data=x, label=y, group=train_baskets)
    model = lgb.train(train_set=train_dataset,
                      params=config.model_param)
    return model


def select_recommendations(candidates, event_type_str, model, config, k=20):
    x = candidates.select(pl.col(config.features)).to_numpy()
    scores = model.predict(x)
    candidates_scored = candidates.select(pl.col(['session', 'aid']))
    candidates_scored = candidates_scored.with_column(
        pl.lit(scores).alias('score'))

    recommendations = candidates_scored.groupby(
        'session').agg(pl.col('aid'))

    recommendations = recommendations.select([(pl.col('session').cast(str) + pl.lit(f'_{event_type_str}')).alias('session_type'), pl.col(
        'aid').apply(lambda x: ' '.join([str(i) for i in x[:k]])).alias('labels')])
    return recommendations
