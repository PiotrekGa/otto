import gc

import pandas as pd
import numpy as np
import numba as nb
import heapq
from tqdm import tqdm

TAIL = 30
PARALLEL = 1024
TOPN = 20
OPS_WEIGTHS = np.array([1.0, 6.0, 3.0])
OP_WEIGHT = 0
TIME_WEIGHT = 1


def load_data(path, fold):
    df = pd.read_parquet(f'{path}{fold}train.parquet')
    df = pd.read_parquet(f'{path}{fold}test.parquet')

    # df = pd.concat([df, df2])
    # del df2

    df = df.astype(np.int64)

    df = df.groupby('session').agg(
        Min=('ts', np.min),
        Count=('ts', np.count_nonzero))
    df.columns = ['start_time', 'length']
    df.reset_index(inplace=True, drop=False)

    # npz = np.load(f"{path}{fold}train.npz")
    # npz_test = np.load(f"{path}{fold}test.npz")
    # aids = np.concatenate([npz['aids'], npz_test['aids']])
    # ts = np.concatenate([npz['ts'], npz_test['ts']])
    # ops = np.concatenate([npz['ops'], npz_test['ops']])

    npz = np.load(f"{path}{fold}test.npz")

    aids = npz['aids']
    ts = npz['ts']
    ops = npz['ops']

    df["idx"] = np.cumsum(df.length) - df.length
    df["end_time"] = df.start_time + ts[df.idx + df.length - 1]

    return df, aids, ts, ops


@nb.jit(nopython=True, cache=True)
def get_single_pairs(pairs, aids, ts, ops, idx, length, start_time, ops_weights, mode):
    max_idx = idx + length
    min_idx = max(max_idx - TAIL, idx)
    for i in range(min_idx, max_idx):
        for j in range(i + 1, max_idx):
            if ts[j] - ts[i] >= 24 * 60 * 60:
                break
            if aids[i] == aids[j]:
                continue
            if mode == OP_WEIGHT:
                w1 = ops_weights[ops[j]]
                w2 = ops_weights[ops[i]]
            elif mode == TIME_WEIGHT:  # FIXME
                w1 = 1 + 3 * (ts[i] + start_time - 1659304800) / \
                    (1662328791 - 1659304800)
                w2 = 1 + 3 * (ts[j] + start_time - 1659304800) / \
                    (1662328791 - 1659304800)
            pairs[(aids[i], aids[j])] = w1
            pairs[(aids[j], aids[i])] = w2


@nb.jit(nopython=True, parallel=True, cache=True)
def get_pairs(aids, ts, ops, row, cnts, ops_weights, mode):
    par_n = len(row)
    pairs = [{(0, 0): 0.0 for _ in range(0)} for _ in range(par_n)]
    for par_i in nb.prange(par_n):
        _, idx, length, start_time = row[par_i]
        get_single_pairs(pairs[par_i], aids, ts, ops, idx,
                         length, start_time, ops_weights, mode)
    for par_i in range(par_n):
        for (aid1, aid2), w in pairs[par_i].items():
            if aid1 not in cnts:
                cnts[aid1] = {0: 0.0 for _ in range(0)}
            cnt = cnts[aid1]
            if aid2 not in cnt:
                cnt[aid2] = 0.0
            cnt[aid2] += w


@nb.jit(nopython=True, cache=True)
def heap_topk(cnt, overwrite, cap):
    q = [(0.0, 0, 0) for _ in range(0)]
    for i, (k, n) in enumerate(cnt.items()):
        if overwrite == 1:
            heapq.heappush(q, (n, i, k))
        else:
            heapq.heappush(q, (n, -i, k))
        if len(q) > cap:
            heapq.heappop(q)
    return [heapq.heappop(q)[2] for _ in range(len(q))][::-1]


@nb.jit(nopython=True, cache=True)
def get_topk(cnts, topk, k):
    for aid1, cnt in cnts.items():
        topk[aid1] = np.array(heap_topk(cnt, 1, k))


def train(df, aids, ts, ops):
    topks = {}
    for mode in [OP_WEIGHT, TIME_WEIGHT]:
        cnts = nb.typed.Dict.empty(
            key_type=nb.types.int64,
            value_type=nb.typeof(nb.typed.Dict.empty(key_type=nb.types.int64, value_type=nb.types.float64)))
        max_idx = len(df)
        for idx in range(0, max_idx, PARALLEL):
            row = df.iloc[idx:min(idx + PARALLEL, max_idx)
                          ][['session', 'idx', 'length', 'start_time']].values
            get_pairs(aids, ts, ops, row, cnts, OPS_WEIGTHS, mode)

        topk = nb.typed.Dict.empty(
            key_type=nb.types.int64,
            value_type=nb.types.int64[:])
        get_topk(cnts, topk, TOPN)
        gc.collect()
        topks[mode] = topk

    return topks


@nb.jit()
def inference_(aids, ops, row, result, topk, test_ops_weights, seq_weight):
    for session, idx, length in row:
        unique_aids = nb.typed.Dict.empty(
            key_type=nb.types.int64, value_type=nb.types.float64)
        cnt = nb.typed.Dict.empty(
            key_type=nb.types.int64, value_type=nb.types.float64)

        candidates = aids[idx:idx + length][::-1]
        candidates_ops = ops[idx:idx + length][::-1]
        for a in candidates:
            unique_aids[a] = 0

        if len(unique_aids) >= 20:
            sequence_weight = np.power(2, np.linspace(
                seq_weight, 1, len(candidates)))[::-1] - 1
            for a, op, w in zip(candidates, candidates_ops, sequence_weight):
                if a not in cnt:
                    cnt[a] = 0
                cnt[a] += w * test_ops_weights[op]
            result_candidates = heap_topk(cnt, 0, 20)
        else:
            result_candidates = list(unique_aids)
            for a in result_candidates:
                if a not in topk:
                    continue
                for b in topk[a]:
                    if b in unique_aids:
                        continue
                    if b not in cnt:
                        cnt[b] = 0
                    cnt[b] += 1
            result_candidates.extend(
                heap_topk(cnt, 0, 20 - len(result_candidates)))
        result[session] = np.array(result_candidates)


@nb.jit()
def inference(aids, ops, row,
              result_clicks, result_buy,
              topk_clicks, topk_buy,
              test_ops_weights):
    inference_(aids, ops, row, result_clicks,
               topk_clicks, test_ops_weights, 0.1)
    inference_(aids, ops, row, result_buy, topk_buy, test_ops_weights, 0.5)


def covisit(path, fold):
    df, aids, ts, ops = load_data(path, fold)
    topks = train(df, aids, ts, ops)

    result_clicks = nb.typed.Dict.empty(
        key_type=nb.types.int64,
        value_type=nb.types.int64[:])
    result_buy = nb.typed.Dict.empty(
        key_type=nb.types.int64,
        value_type=nb.types.int64[:])
    df_test = pd.read_parquet(f'{path}{fold}test.parquet')

    for idx in tqdm(range(len(df) - len(df_test), len(df), PARALLEL)):
        row = df.iloc[idx:min(idx + PARALLEL, len(df))
                      ][['session', 'idx', 'length']].values
        inference(aids, ops, row, result_clicks, result_buy,
                  topks[TIME_WEIGHT], topks[OP_WEIGHT], OP_WEIGHT)

    subs = []
    op_names = ["clicks", "carts", "orders"]
    for result, op in zip([result_clicks, result_buy, result_buy], op_names):

        sub = pd.DataFrame(
            {"session_type": result.keys(), "labels": result.values()})
        sub.session_type = sub.session_type.astype(str) + f"_{op}"
        sub.labels = sub.labels.apply(lambda x: " ".join(x.astype(str)))
        subs.append(sub)

    sub = pd.concat(subs).reset_index(drop=True)

    return sub
