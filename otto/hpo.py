import polars as pl
from tqdm import tqdm

from candidates import generate_candidates
from features import add_labels, add_featues
from rerank_model import train_rerank_model, select_recommendations
from evaluate import evaluate
import optuna
import joblib


def cross_val(candidates_train, candidates_valid, config):
    fold = ['valid2__', 'valid3__']

    model_orders = train_rerank_model(candidates_train, 'y_orders', config)

    reco_orders = select_recommendations(
        candidates_valid, 'orders', model_orders, config)

    reco = pl.concat([reco_orders, reco_orders.with_column(pl.col('session_type').str.replace(
        'orders', 'carts')), reco_orders.with_column(pl.col('session_type').str.replace('orders', 'clicks'))])

    reco.write_csv('hpo_temp.csv')

    score = evaluate(
        f'{config.data_path}raw/{fold[1]}test_labels.jsonl', 'hpo_temp.csv')

    return score['orders']


def objective(trial):

    joblib.dump(study, 'study.pkl')

    learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.1, log=True)
    num_leaves = trial.suggest_int('num_leaves', 5, 100, log=True)
    feature_fraction = trial.suggest_float('feature_fraction', 0.2, 1.0)
    bagging_fraction = trial.suggest_float('bagging_fraction', 0.2, 1.0)
    min_child_samples = trial.suggest_int(
        'min_child_samples', 10, 100, log=True)

    class CONFIG:
        data_path = '../data/'

        max_negative_candidates = trial.suggest_int(
            'max_negative_candidates', 10, 50, log=True)

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
            'session_interaction_cnt',
            'session_interaction_last_time',
            'click_interaction_cnt',
            'click_interaction_last_time',
            'cart_interaction_cnt',
            'cart_interaction_last_time',
            'order_interaction_cnt',
            'order_interaction_last_time',
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
                       'metric': 'map',
                       'verbose': -1,
                       'n_jobs': -1,
                       'num_iterations': 500,
                       'learning_rate': learning_rate,
                       'num_leaves': num_leaves,
                       'feature_fraction': feature_fraction,
                       'bagging_fraction': bagging_fraction,
                       'min_child_samples': min_child_samples,
                       'feature_pre_filter': False
                       }
    print(candidates_train.shape, candidates_valid.shape)
    return cross_val(candidates_train, candidates_valid, CONFIG)


if __name__ == '__main__':

    fold = ['valid2__', 'valid3__']

    class config:
        data_path = '../data/'

    candidates_train = generate_candidates(fold[0], config)
    candidates_train = add_labels(candidates_train, fold[0], config)
    candidates_train = add_featues(candidates_train, fold[0], config)

    candidates_valid = generate_candidates(fold[1], config)
    candidates_valid = add_labels(candidates_valid, fold[1], config)
    candidates_valid = add_featues(candidates_valid, fold[1], config)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=200)
    joblib.dump(study, 'study.pkl')
