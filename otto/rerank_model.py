import polars as pl
from lightgbm import LGBMRanker, LGBMClassifier


def train_rerank_model(candidates, train_column, config):
    y = candidates.select(pl.col(train_column)).to_numpy().ravel()
    x = candidates.select(pl.col(config.features)).to_numpy()

    model = LGBMClassifier()
    model.fit(x, y)
    return model


def score_candidates(candidates, model, config, topk=20):
    x = candidates.select(pl.col(config.features)).to_numpy()
    scores = model.predict_proba(x)[:, 1]
    candidates_scored = candidates.select(pl.col(['session', 'aid']))
    candidates_scored = candidates_scored.with_column(
        pl.lit(scores).alias('score'))
    return score_candidates
