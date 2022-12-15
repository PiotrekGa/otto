import gc
import polars as pl
from tqdm import tqdm

from candidates import generate_candidates
from features import add_labels, add_featues
from rerank_model import train_rerank_model, select_recommendations, select_perfect_recommendations
from evaluate import evaluate


class CONFIG:
    score_perfect = True
    data_path = '../data/'
    submission_name = 'submission2'
    folds = [['valid2__', 'valid3__']]  # , ['valid3__', '']]

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
        'mf1_clicks',
        'mf1_carts',
        'mf1_orders',
        # 'w2v_09_clicks',
        # 'w2v_09_carts',
        # 'w2v_09_orders',
        'session_interaction_cnt',
        'session_interaction_last_time',
        # 'session_last_weekday',
        # 'session_last_time_of_day',
        'click_interaction_cnt',
        'click_interaction_last_time',
        # 'click_last_weekday',
        # 'click_last_time_of_day',
        'cart_interaction_cnt',
        'cart_interaction_last_time',
        # 'cart_last_weekday',
        # 'cart_last_time_of_day',
        'order_interaction_cnt',
        'order_interaction_last_time',
        # 'order_last_weekday',
        # 'order_last_time_of_day',
        'aid_max_ts',
        'aid_min_ts',
        'aid_cnt',
        'aid_cart_max_ts',
        'aid_cart_min_ts',
        'aid_cart_cnt',
        'aid_order_max_ts',
        'aid_order_min_ts',
        'aid_order_cnt',
        'click_to_cart',
        'click_to_order',
        'cart_to_order']
    model_param = {'objective': 'lambdarank',
                   'lambdarank_truncation_level': 15,
                   'verbose': -1,
                   'n_jobs': -1}


def main(config):
    scores = []
    scores_perfect = []
    for fold in tqdm(config.folds):
        print('FOLD', fold[0], fold[1])
        candidates_train = generate_candidates(fold[0], config)
        candidates_train = add_labels(candidates_train, fold[0], config)
        candidates_train = add_featues(candidates_train, fold[0], config)

        model_clicks = train_rerank_model(candidates_train, 'y_clicks', config)
        model_carts = train_rerank_model(candidates_train, 'y_carts', config)
        model_orders = train_rerank_model(candidates_train, 'y_orders', config)

        del candidates_train
        gc.collect()

        candidates_valid = generate_candidates(fold[1], config)
        if len(fold[1]) > 0:
            candidates_valid = add_labels(candidates_valid, fold[1], config)
        candidates_valid = add_featues(candidates_valid, fold[1], config)

        gc.collect()
        reco_clicks = select_recommendations(
            candidates_valid, 'clicks', model_clicks, config)
        reco_carts = select_recommendations(
            candidates_valid, 'carts', model_carts, config)
        reco_orders = select_recommendations(
            candidates_valid, 'orders', model_orders, config)

        reco = pl.concat([reco_clicks, reco_carts, reco_orders])
        reco.write_csv(f'{fold[1]}{config.submission_name}.csv')

        del reco_clicks, reco_carts, reco_orders
        gc.collect()

        if config.score_perfect and len(fold[1]) > 0:
            reco_clicks = select_perfect_recommendations(
                candidates_valid, 'clicks')
            reco_carts = select_perfect_recommendations(
                candidates_valid, 'carts')
            reco_orders = select_perfect_recommendations(
                candidates_valid, 'orders')

            reco_perfect = pl.concat([reco_clicks, reco_carts, reco_orders])
            reco_perfect.write_csv(
                f'{fold[1]}{config.submission_name}_perfect.csv')
            del reco_clicks, reco_carts, reco_orders
            gc.collect()

        if len(fold[1]) > 0:
            score = evaluate(
                f'{config.data_path}raw/{fold[1]}test_labels.jsonl', f'{fold[1]}{config.submission_name}.csv')
            print(f"Scores: {score}")
            scores.append(score)

            if config.score_perfect:
                score_perfect = evaluate(
                    f'{config.data_path}raw/{fold[1]}test_labels.jsonl', f'{fold[1]}{config.submission_name}_perfect.csv')
                print(f"Scores perfect: {score_perfect}")
                scores_perfect.append(score_perfect)

    if len(scores) > 0:
        scores = pl.DataFrame(scores).mean()
        scores_perfect = None
        if config.score_perfect:
            scores_perfect = pl.DataFrame(scores_perfect).mean()
        return scores, scores_perfect
    else:
        return None


if __name__ == '__main__':
    scores, scores_perfect = main(CONFIG)
    print(scores)
    print(scores_perfect)

    # {'clicks': 0.5405462010808024, 'carts': 0.4234126178005847,
    #     'orders': 0.6609028481368907, 'total': 0.5776201143303901}
