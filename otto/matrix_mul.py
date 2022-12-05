import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm
from copy import deepcopy
import gc
from joblib import Parallel, delayed


def matrix_mult_pred(path, prefix, event_to_predict, filter_event=None):
    # df = pd.concat([pd.read_parquet(f'{path}{prefix}train.parquet'),
    #                pd.read_parquet(f'{path}{prefix}test.parquet')])
    df = pd.read_parquet(f'{path}{prefix}test.parquet')
    if filter_event:
        df = df.loc[df.type == filter_event, :]
    df['session_next'] = df.session.shift(1)
    df['aid_next'] = df.aid.shift(1)
    df['type_next'] = df.type.shift(1)
    df.dropna(inplace=True)
    df.aid_next = df.aid_next.astype(np.int32)
    df.session_next = df.session_next.astype(np.int32)
    df.type_next = df.type_next.astype(np.int8)
    df = df.loc[df.type_next == event_to_predict, :]
    df = df.loc[df.session_next == df.session, :]
    xx = df.groupby(['aid', 'aid_next']).count()['ts'].reset_index()
    new_prod_id = 1855603
    xx = pd.concat([xx, pd.DataFrame(
        np.array([[new_prod_id, new_prod_id, 1]]), columns=['aid', 'aid_next', 'ts'])])
    xx.reset_index(inplace=True, drop=True)
    matrix_shape = new_prod_id + 1
    x = sparse.coo_matrix((xx.ts, (xx.aid, xx.aid_next)),
                          shape=(matrix_shape, matrix_shape))
    x = x.tocsr()
    x.data = x.data / \
        np.repeat(np.add.reduceat(x.data, x.indptr[:-1]), np.diff(x.indptr))
    x1 = deepcopy(x)
    for _ in tqdm(range(1)):
        x1 = x1.dot(x)
        x1.data = x1.data / \
            np.repeat(np.add.reduceat(
                x1.data, x1.indptr[:-1]), np.diff(x1.indptr))
    del x, df, xx
    gc.collect()
    return x1


def make_pred(matrix_dict, idx):
    preds = np.array(matrix_dict[idx, :].todense())[0, :]
    preds = np.argpartition(preds, -20)[-20:]
    preds = ' '.join([str(i) for i in preds])
    return preds


def make_sub(df, matrix_dict, event_name):
    prediction_dict = Parallel(n_jobs=8)(delayed(make_pred)(
        matrix_dict, idx) for idx in tqdm(range(1855603)))
    prediction_dict = pd.Series(prediction_dict)
    df['labels'] = df.aid.map(prediction_dict)
    df.reset_index(drop=False, inplace=True)
    df['session_type'] = df.session.astype(str) + '_' + event_name
    df = df.loc[:, ['session_type', 'labels']]
    return df


if __name__ == '__main__':

    PREFIX = ''

    class CONFIG:
        path = '../data/raw/'
        prefix = PREFIX
        filter_event_clicks = None
        filter_event_carts = None
        filter_event_orders = None

        sub_name = f'{prefix}submission_mat_mul_0'

    df = pd.read_parquet(
        f'{CONFIG.path}{CONFIG.prefix}test.parquet')

    clicks_dict = matrix_mult_pred(
        CONFIG.path, CONFIG.prefix, 0, CONFIG.filter_event_clicks)
    if CONFIG.filter_event_clicks:
        df_clicks = df.loc[df.type == CONFIG.filter_event_clicks]
    else:
        df_clicks = df.copy()
    df_clicks = df_clicks.groupby('session')[['aid']].last()
    sub_clicks = make_sub(df_clicks, clicks_dict, 'clicks')
    del clicks_dict, df_clicks

    carts_dict = matrix_mult_pred(
        CONFIG.path, CONFIG.prefix, 1, CONFIG.filter_event_carts)
    if CONFIG.filter_event_carts:
        df_carts = df.loc[df.type == CONFIG.filter_event_carts]
    else:
        df_carts = df.copy()
    df_carts = df_carts.groupby('session')[['aid']].last()
    sub_carts = make_sub(df_carts, carts_dict, 'carts')
    del carts_dict, df_carts

    orders_dict = matrix_mult_pred(
        CONFIG.path, CONFIG.prefix, 2, CONFIG.filter_event_orders)
    if CONFIG.filter_event_orders:
        df_orders = df.loc[df.type == CONFIG.filter_event_orders]
    else:
        df_orders = df.copy()
    df_orders = df_orders.groupby('session')[['aid']].last()
    sub_orders = make_sub(df_orders, orders_dict, 'orders')
    del orders_dict, df_orders

    sub = pd.concat([sub_clicks, sub_carts, sub_orders])
    sub.to_csv(f'{CONFIG.sub_name}.csv', index=False)

    # class CONFIG:
    #     path = '../data/raw/'
    #     prefix = PREFIX
    #     filter_event_clicks = 0
    #     filter_event_carts = 1
    #     filter_event_orders = 2

    #     sub_name = f'{prefix}submission_mat_mul_1'

    # df = pd.read_parquet(
    #     f'{CONFIG.path}{CONFIG.prefix}test.parquet')

    # clicks_dict = matrix_mult_pred(
    #     CONFIG.path, CONFIG.prefix, 0, CONFIG.filter_event_clicks)
    # if CONFIG.filter_event_clicks:
    #     df_clicks = df.loc[df.type == CONFIG.filter_event_clicks]
    # else:
    #     df_clicks = df.copy()
    # df_clicks = df_clicks.groupby('session')['aid'].apply(list)
    # sub_clicks = make_sub(df_clicks, clicks_dict, 'clicks')
    # del clicks_dict, df_clicks

    # carts_dict = matrix_mult_pred(
    #     CONFIG.path, CONFIG.prefix, 1, CONFIG.filter_event_carts)
    # if CONFIG.filter_event_carts:
    #     df_carts = df.loc[df.type == CONFIG.filter_event_carts]
    # else:
    #     df_carts = df.copy()
    # df_carts = df_carts.groupby('session')['aid'].apply(list)
    # sub_carts = make_sub(df_carts, carts_dict, 'carts')
    # del carts_dict, df_carts

    # orders_dict = matrix_mult_pred(
    #     CONFIG.path, CONFIG.prefix, 2, CONFIG.filter_event_orders)
    # if CONFIG.filter_event_orders:
    #     df_orders = df.loc[df.type == CONFIG.filter_event_orders]
    # else:
    #     df_orders = df.copy()
    # df_orders = df_orders.groupby('session')['aid'].apply(list)
    # sub_orders = make_sub(df_orders, orders_dict, 'orders')
    # del orders_dict, df_orders

    # sub = pd.concat([sub_clicks, sub_carts, sub_orders])
    # sub.to_csv(f'{CONFIG.sub_name}.csv', index=False)
