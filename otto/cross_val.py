import gc
import polars as pl
from tqdm import tqdm
from time import sleep
import joblib

from candidates import generate_candidates
from features import add_labels, add_featues
from rerank_model import train_rerank_model, select_recommendations, select_perfect_recommendations, sample_candidates
from evaluate import evaluate


class CONFIG:
    score_perfect = True
    data_path = '../data/'
    submission_name = 'submission'
    # folds = [['valid2__', 'valid3__'], ['valid3__', '']]
    folds = [['valid3__', '']]

    sample_size = None
    max_negative_candidates = 50

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
        'civisit3',
        'civisit4',
        'civisit5',
        'covisit6',
        'covisit7',

        'covisit8',
        'covisit9',
        'covisit10',
        'covisit11',
        'covisit12',
        'covisit13',

        'tg_covisit1',

        'recbole_clicks',
        'recbole_clicks2',

        'cleora_emde',

        'ses2ses3',
        'ses2ses4',
        'ses2ses5',

        'bpr_score',

        'covisit_score1',
        'covisit_score2',
        'covisit_score3',
        'covisit_score4',
        'covisit_score5',

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
        'order_session_aid_cnt_m',

        'leak_top_day_clicks',
        'leak_top_day_clicks_cnt',
        'leak_top_day_carts',
        'leak_top_day_carts_cnt',
        'leak_top_day_orders',
        'leak_top_day_orders_cnt',

        'leak_top_days_clicks',
        'leak_top_days_clicks_cnt',
        'leak_top_days_carts',
        'leak_top_days_carts_cnt',
        'leak_top_days_orders',
        'leak_top_days_orders_cnt',

        'bpr_cands',

    ]
    model_param = {'objective': 'lambdarank',
                   'lambdarank_truncation_level': 15,
                   'verbose': -1,
                   'n_jobs': -1,
                   'boosting_type': 'dart',
                   'num_boost_round': 750}


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

        models = {}
        for target in ['y_clicks', 'y_carts', 'y_orders']:

            candidates_target = sample_candidates(
                candidates_train, target, config)

            candidates_target = add_featues(candidates_target, fold[0], config)
            gc.collect()
            if target == 'y_orders':
                candidates_target.write_parquet(
                    f'{fold[0]}{target}_train_data.parquet')
            models[target] = train_rerank_model(
                candidates_target, target, config)
            del candidates_target
            gc.collect()

        del candidates_train
        gc.collect()

        joblib.dump(models, 'models.pkl')
        sleep(1)
        models = joblib.load('models.pkl')

        candidates_valid = generate_candidates(fold[1], config)
        if len(fold[1]) > 0:
            candidates_valid = add_labels(candidates_valid, fold[1], config)
            gc.collect()
        candidates_valid = add_featues(candidates_valid, fold[1], config)

        gc.collect()

        reco_clicks = select_recommendations(
            candidates_valid, 'clicks', models['y_clicks'], config)
        gc.collect()
        reco_carts = select_recommendations(
            candidates_valid, 'carts', models['y_carts'], config)
        gc.collect()
        reco_orders = select_recommendations(
            candidates_valid, 'orders', models['y_orders'], config)
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

# Scores: {'clicks': 0.54176313948243, 'carts': 0.42397258459119774, 'orders': 0.6596227143339606, 'total': 0.5771417179259787}
# evaluating solution
# Scores perfect: {'clicks': 0.6614316258987507, 'carts': 0.5287217035708923, 'orders': 0.7197021598237404, 'total': 0.656580969555387}
