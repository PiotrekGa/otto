import polars as pl
from tqdm import tqdm

from candidates import generate_candidates
from features import add_labels, add_featues
from rerank_model import train_rerank_model, select_recommendations
from evaluate import evaluate


class CONFIG:
    data_path = '../data/'
    submission_name = 'submission'
    folds = [['valid1__', 'valid2__'], [
        'valid2__', 'valid3__'], ['valid3__', '']]

    max_negative_candidates = 20
    features = [
        'clicked_in_session',
        'carted_in_session',
        'ordered_in_session',
        'cart_without_order',
        'click_without_order',
        'click_without_cart',
        'covisit1_clicks',
        'covisit1_carts',
        'session_interaction_cnt',
        'session_interaction_last_time',
        'click_interaction_cnt',
        'click_interaction_last_time',
        'cart_interaction_cnt',
        'cart_interaction_last_time',
        'order_interaction_cnt',
        'order_interaction_last_time']
    model_param = {'objective': 'lambdarank',
                   'lambdarank_truncation_level': 15,
                   'verbose': -1}


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

        reco_clicks = select_recommendations(
            candidates_valid, 'clicks', model_clicks, config)
        reco_carts = select_recommendations(
            candidates_valid, 'carts', model_carts, config)
        reco_orders = select_recommendations(
            candidates_valid, 'orders', model_orders, config)

        reco = pl.concat([reco_clicks, reco_carts, reco_orders])
        reco.write_csv(f'{fold[1]}{config.submission_name}.csv')

        if len(fold[1]) > 0:
            score = evaluate(
                f'{config.data_path}raw/{fold[1]}test_labels.jsonl', f'{fold[1]}{config.submission_name}.csv')
            scores.append(score)

    if len(scores) > 0:
        scores = pl.DataFrame(scores).mean()
        return scores
    else:
        return None


if __name__ == '__main__':
    scores = main(CONFIG)
    print(scores)
