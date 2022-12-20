import gc
import polars as pl
from tqdm import tqdm
from time import sleep

from candidates import generate_candidates
from features import add_labels, add_featues
from rerank_model import train_rerank_model, select_recommendations, select_perfect_recommendations
from evaluate import evaluate


class CONFIG:
    score_perfect = True
    data_path = '../data/'
    submission_name = 'submission'
    folds = [['valid2__', 'valid3__'], ['valid3__', '']]

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
        'w2v_09_clicks',
        'w2v_09_carts',
        'w2v_09_orders',
        'w2v_01_clicks',
        'w2v_01_carts',
        'w2v_01_orders',
        'w2v_35_clicks',
        'w2v_35_carts',
        'w2v_35_orders',
        'civisit2',

        'session_interaction_cnt',
        'session_interaction_last_time',
        'session_last_weekday',
        'session_last_time_of_day',
        'click_interaction_cnt',
        'click_interaction_last_time',
        'click_last_weekday',
        'click_last_time_of_day',
        'cart_interaction_cnt',
        'cart_interaction_last_time',
        'cart_last_weekday',
        'cart_last_time_of_day',

        'order_interaction_cnt',
        'order_interaction_last_time',
        'order_last_weekday',
        'order_last_time_of_day',

        'aid_max_ts',
        'aid_min_ts',
        'aid_cnt',
        'aid_sess_cnt',

        'aid_click_max_ts',
        'aid_click_min_ts',
        'aid_click_cnt',
        'aid_sess_click_cnt',

        'aid_cart_max_ts',
        'aid_cart_min_ts',
        'aid_cart_cnt',
        'aid_sess_cart_cnt',

        'aid_order_max_ts',
        'aid_order_min_ts',
        'aid_order_cnt',
        'aid_sess_order_cnt',

        'click_to_cart',
        'click_to_order',
        'cart_to_order',

        'session_cnt',
        'session_cnt_distinct',
        'session_events_per_aid',
        'click_cnt',
        'click_cnt_distinct',
        'click_events_per_aid',
        'cart_cnt',
        'cart_cnt_distinct',
        'cart_events_per_aid',
        'order_cnt',
        'order_cnt_distinct',
        'order_events_per_aid',

        'session_session_aid_cnt_m',
        'click_session_aid_cnt_m',
        'cart_session_aid_cnt_m',
        'order_session_aid_cnt_m'


    ]
    model_param = {'objective': 'lambdarank',
                   'lambdarank_truncation_level': 15,
                   'verbose': -1,
                   'n_jobs': -1}


def main(config):
    sleep(1)
    scores = []
    scores_perfect = []
    for fold in tqdm(config.folds):
        print('FOLD', fold[0], fold[1])
        candidates_train = generate_candidates(fold[0], config)
        gc.collect()
        candidates_train = add_labels(candidates_train, fold[0], config)
        gc.collect()
        candidates_train = add_featues(candidates_train, fold[0], config)
        gc.collect()

        model_clicks = train_rerank_model(candidates_train, 'y_clicks', config)
        gc.collect()
        model_carts = train_rerank_model(candidates_train, 'y_carts', config)
        gc.collect()
        model_orders = train_rerank_model(candidates_train, 'y_orders', config)

        del candidates_train
        gc.collect()

        candidates_valid = generate_candidates(fold[1], config)
        if len(fold[1]) > 0:
            candidates_valid = add_labels(candidates_valid, fold[1], config)
            gc.collect()
        candidates_valid = add_featues(candidates_valid, fold[1], config)
        gc.collect()

        reco_clicks = select_recommendations(
            candidates_valid, 'clicks', model_clicks, config)
        gc.collect()
        reco_carts = select_recommendations(
            candidates_valid, 'carts', model_carts, config)
        gc.collect()
        reco_orders = select_recommendations(
            candidates_valid, 'orders', model_orders, config)
        gc.collect()
        reco = pl.concat([reco_clicks, reco_carts, reco_orders])
        reco.write_csv(f'{fold[1]}{config.submission_name}.csv')

        del reco_clicks, reco_carts, reco_orders
        gc.collect()

        if config.score_perfect and len(fold[1]) > 0:
            reco_clicks = select_perfect_recommendations(
                candidates_valid, 'clicks')
            gc.collect()
            reco_carts = select_perfect_recommendations(
                candidates_valid, 'carts')
            gc.collect()
            reco_orders = select_perfect_recommendations(
                candidates_valid, 'orders')
            gc.collect()

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
            gc.collect()
            if config.score_perfect:
                score_perfect = evaluate(
                    f'{config.data_path}raw/{fold[1]}test_labels.jsonl', f'{fold[1]}{config.submission_name}_perfect.csv')
                print(f"Scores perfect: {score_perfect}")
                scores_perfect.append(score_perfect)
                gc.collect()
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

# Scores: {'clicks': 0.5323509984073355, 'carts': 0.41837644959857273, 'orders': 0.6570849616442321, 'total': 0.5729990117068446}
# Scores perfect: {'clicks': 0.5957445706710365, 'carts': 0.47546833184656556, 'orders': 0.6904456345185829, 'total': 0.616482337332223}
