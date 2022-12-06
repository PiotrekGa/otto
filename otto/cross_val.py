import polars as pl
from tqdm import tqdm

from candidates import generate_candidates
from features import add_labels, add_featues
from rerank_model import train_rerank_model, score_candidates


class CONFIG:
    data_path = '../data/'
    folds = [['valid1__', 'valid2__'], [
        'valid2__', 'valid3__'], ['valid3__', '']]
    features = [
        'clicked_in_session',
        'carted_in_session',
        'ordered_in_session',
        'session_interaction_cnt',
        'session_interaction_last_time',
        'click_interaction_cnt',
        'click_interaction_last_time',
        'cart_interaction_cnt',
        'cart_interaction_last_time',
        'order_interaction_cnt',
        'order_interaction_last_time']


def main(config):
    scores = []
    for fold in tqdm(config.folds):
        print('FOLD', fold[0], fold[1])
        candidates_train = generate_candidates(fold[0], config)
        candidates_train = add_labels(candidates_train, fold[0], config)
        candidates_train = add_featues(candidates_train, fold[0], config)

        model_clicks = train_rerank_model(candidates_train, 'y_clicks', config)
        model_carts = train_rerank_model(candidates_train, 'y_carts', config)
        model_orders = train_rerank_model(candidates_train, 'y_orders', config)

        candidates_valid = generate_candidates(fold[1], config)
        if len(fold[1]) > 0:
            candidates_valid = add_labels(candidates_valid, fold[1], config)
        candidates_valid = add_featues(candidates_valid, fold[1], config)

        scored_candidates_clicks = score_candidates(
            candidates_valid, model_clicks, config)
        scored_candidates_carts = score_candidates(
            candidates_valid, model_carts, config)
        scored_candidates_orders = score_candidates(
            candidates_valid, model_orders, config)


if __name__ == '__main__':
    main(CONFIG)
