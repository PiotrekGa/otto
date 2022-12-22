import polars as pl
from candidates import CovisitMaster
import gc
import optuna
from evaluate import evaluate
import joblib


def objective(trial):
    gc.collect()
    click_weight = 1
    cart_weight = trial.suggest_float('cart_weight', 0, 50)
    order_weight = trial.suggest_float('order_weight', 0, 50)
    days_back = trial.suggest_int("days_back", 1, 21)
    before_time = 0
    print('before_time', before_time)
    before_time = before_time * 60 * 60
    after_time = trial.suggest_int("after_time", 0, 48)
    print('after_time', after_time)
    after_time = after_time * 60 * 60
    left_types = [0, 1, 2]
    right_types = trial.suggest_categorical(
        'right_types', ['1', '2', '0', '12', '01', '02', '012'])
    print('right_types', right_types)
    right_types = [int(i) for i in right_types]
    time_weight_coef = trial.suggest_float('time_weight_coef', 0, 10)
    normalize = trial.suggest_categorical('normalize', [False, True])
    session_hist = trial.suggest_int("session_hist", 1, 50)
    joblib.dump(study, f'{event_optimized}_study.pkl')

    print('cart_weight', cart_weight)
    print('order_weight', order_weight)
    print('days_back', days_back)
    print('time_weight_coef', time_weight_coef)
    print('normalize', normalize)
    print('session_hist', session_hist)

    cm = CovisitMaster(fold=fold, name='optuna', data_path='../data/', max_cands=20, type_weight={0: click_weight, 1: cart_weight, 2: order_weight},
                       days_back=days_back, before_time=before_time, after_time=after_time, left_types=left_types, right_types=right_types,
                       time_weight_coef=time_weight_coef, normalize=normalize, session_hist=session_hist)

    reco = cm.prepare_candidates(return_df=True)
    print('cands ready')
    reco = reco.groupby(
        'session').agg(pl.col('aid'))

    reco_main = reco.select([(pl.col('session').cast(str) + pl.lit(f'_{event_optimized}')).alias('session_type'), pl.col(
        'aid').apply(lambda x: ' '.join([str(i) for i in x])).alias('labels')])
    print('reco_main ready')
    if event_optimized == 'orders':
        reco_rest1 = reco_main.with_column(
            pl.col('session_type').str.replace('orders', 'clicks'))
        reco_rest = reco_main.with_column(
            pl.col('session_type').str.replace('orders', 'carts'))
    elif event_optimized == 'carts':
        reco_rest1 = reco_main.with_column(
            pl.col('session_type').str.replace('carts', 'clicks'))
        reco_rest = reco_main.with_column(
            pl.col('session_type').str.replace('carts', 'orders'))
    elif event_optimized == 'clicks':
        reco_rest1 = reco_main.with_column(
            pl.col('session_type').str.replace('clicks', 'carts'))
        reco_rest = reco_main.with_column(
            pl.col('session_type').str.replace('clicks', 'orders'))
    reco_main = pl.concat([reco_main, reco_rest1, reco_rest])
    reco_main.write_csv('optuna.csv')
    del reco, reco_main, reco_rest, reco_rest1
    scores = evaluate(f'../data/raw/{fold}test_labels.jsonl', 'optuna.csv')
    print(scores)
    gc.collect()
    return scores[event_optimized]


if __name__ == '__main__':
    fold = 'valid2__'
    event_optimized = 'orders'
    # study = optuna.create_study(direction="maximize")
    study = joblib.load(f'{event_optimized}_study.pkl')
    study.enqueue_trial({'cart_weight': 6,
                         'order_weight': 3,
                         'days_back': 14,
                         'after_time': 24,
                         'right_types': '012',
                         'time_weight_coef': 3,
                         'normalize': True,
                         'session_hist': 30})
    study.optimize(objective, timeout=int(10.5*60*60))
    joblib.dump(study, f'{event_optimized}_study.pkl')
