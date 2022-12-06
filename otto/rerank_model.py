import polars as pl
import lightgbm as lgb


def train_rerank_model(candidates, train_column, config):

    candidates = candidates.sort(by='session')

    non_neg = candidates.groupby('session').agg(
        pl.col(train_column).max().alias('is_positive'))
    non_neg = non_neg.filter(pl.col('is_positive') >
                             0).select(pl.col('session'))
    candidates = candidates.join(non_neg, on='session', how='inner')
    del non_neg

    y = candidates.select(pl.col(train_column)).to_numpy().ravel()
    x = candidates.select(pl.col(config.features)).to_numpy()
    train_baskets = candidates.groupby(['session']).agg(
        pl.col('aid').count().alias('basket'))
    train_baskets = train_baskets.select(pl.col('basket'))
    train_baskets = train_baskets.to_numpy().ravel()

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
