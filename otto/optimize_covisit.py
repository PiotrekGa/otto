import polars as pl
from candidates import CovisitMaster
import gc
import optuna
import joblib


def f1_score(preds, fold, path, event_optimized=None):
    weight_dict = {0: .1, 1: .3, 2: .6}
    event_dict = {'clicks': 0, 'carts': 1, 'orders': 2}
    if event_optimized is not None:
        event_optimized = event_dict[event_optimized]
    max_pred_len = preds.select(pl.col('aid').apply(len)).max()[0, 0]
    print(f'EVALUATION FOR {max_pred_len} PREDICTIONS')

    labels = pl.read_parquet(f'{path}{fold}test_labels.parquet')
    labels = labels.groupby(['session', 'type']).agg_list()
    labels.columns = ['session', 'type', 'labels']
    labels = labels.join(preds, on='session', how='left')
    labels = labels.with_column(pl.when(pl.col('aid').is_null()).then([]).otherwise(pl.col('aid'))
                                .keep_name())
    labels = labels.with_column(pl.struct(['labels', 'aid']).apply(
        lambda x: sum([1 for i in x['labels'] if i in x['aid']])).alias('met1'))
    labels = labels.with_column(pl.col('labels').apply(
        lambda x: min(max_pred_len, len(x))).alias('met2'))
    labels = labels.with_column(pl.struct(['labels', 'aid']).apply(
        lambda x: sum([1 for i in x['aid'] if i in x['labels']])).alias('met3'))
    labels = labels.with_column(
        pl.col('aid').apply(lambda x: len(x)).alias('met4'))

    results = labels.select(
        ['type', 'met1', 'met2', 'met3', 'met4']).groupby('type').sum()
    results = results.with_column(
        (pl.col('met1') / pl.col('met2')).alias('recall'))
    results = results.with_column(
        (pl.col('met3') / pl.col('met4')).alias('precision'))
    if event_optimized is None:
        results = results.with_column(pl.col('type').apply(
            lambda x: weight_dict[x]).alias('weight'))
        results = results.select(
            [pl.col('recall') * pl.col('weight'), pl.col('precision') * pl.col('weight')]).sum()
    else:
        results = results.filter(pl.col('type') == event_optimized)
        results = results.select(pl.col(['recall', 'precision']))
    print(results)
    return results[0, 0]
    # return (2 * results[0, 0] * results[0, 1]) / (results[0, 0] + results[0, 1])


def objective(trial):
    gc.collect()
    click_weight = 1
    cart_weight = trial.suggest_float('cart_weight', 0, 50)
    order_weight = trial.suggest_float('order_weight', 0, 50)
    days_back = trial.suggest_int("days_back", 1, 21)
    before_time = trial.suggest_int("before_time", 0, 48)
    print('before_time', before_time)
    before_time = before_time * 60 * 60
    after_time = trial.suggest_int("after_time", 0, 48)
    print('after_time', after_time)
    after_time = after_time * 60 * 60
    left_types = trial.suggest_categorical(
        'left_types', ['1', '2', '0', '12', '01', '02', '012'])
    print('left_types', left_types)
    left_types = [int(i) for i in left_types]
    right_types = trial.suggest_categorical(
        'right_types', ['1', '2', '0', '12', '01', '02', '012'])
    print('right_types', right_types)
    right_types = [int(i) for i in right_types]
    time_weight_coef = trial.suggest_float('time_weight_coef', 0, 10)
    normalize = trial.suggest_categorical('normalize', [False, True])
    session_hist = trial.suggest_int("session_hist", 1, 30)
    if event_optimized is not None:
        joblib.dump(study, f'{event_optimized}_study.pkl')
    else:
        joblib.dump(study, 'study.pkl')

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

    try:
        score = f1_score(reco, fold, '../data/raw/', )
        print(score)
    except:
        score = -1
    gc.collect()
    return score


if __name__ == '__main__':
    fold = 'valid2__'
    event_optimized = None
    study = joblib.load('study.pkl')
    # study = optuna.create_study(direction="maximize")
    # study.enqueue_trial({'cart_weight': 6,
    #                      'order_weight': 3,
    #                      'days_back': 14,
    #                      'before_time': 0,
    #                      'after_time': 24,
    #                      'left_types': '012',
    #                      'right_types': '012',
    #                      'time_weight_coef': 3,
    #                      'normalize': True,
    #                      'session_hist': 30})
    study.optimize(objective, timeout=int(15*60*60))
    if event_optimized is not None:
        joblib.dump(study, f'{event_optimized}_study.pkl')
    else:
        joblib.dump(study, 'study.pkl')
