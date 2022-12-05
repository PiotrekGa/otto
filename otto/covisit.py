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
TEST_OPS_WEIGTHS = np.array([1.0, 6.0, 3.0])


def load_data(path, file_name):
    df = pd.read_parquet(f'{path}{file_name}.parquet')
    df = df.groupby('session').agg(
        Min=('ts', np.min),
        Count=('ts', np.count_nonzero))
    df.columns = ['start_time', 'length']
    df.reset_index(inplace=True, drop=False)

    npz = np.load(f"{path}{file_name}.npz")
    aids = npz['aids']
    ts = npz['ts']
    ops = npz['ops']

    df["idx"] = np.cumsum(df.length) - df.length
    df["end_time"] = df.start_time + ts[df.idx + df.length - 1]

    return df, aids, ts, ops

# get pair dict {(aid1, aid2): weight} for each session
# The maximum time span between two points is 1 day = 24 * 60 * 60 sec


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
            elif mode == TIME_WEIGHT:
                w1 = 1 + 3 * (ts[i] + start_time - 1659304800) / \
                    (1662328791 - 1659304800)
                w2 = 1 + 3 * (ts[j] + start_time - 1659304800) / \
                    (1662328791 - 1659304800)
            pairs[(aids[i], aids[j])] = w1
            pairs[(aids[j], aids[i])] = w2

# get pair dict of each session in parallel
# merge pairs into a nested dict format (cnt)


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

# util function to get most common keys from a counter dict using min-heap
# overwrite == 1 means the later item with equal weight is more important
# otherwise, means the former item with equal weight is more important
# the result is ordered from higher weight to lower weight


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

# save top-k aid2 for each aid1's cnt


@nb.jit(nopython=True, cache=True)
def get_topk(cnts, topk, k):
    for aid1, cnt in cnts.items():
        topk[aid1] = np.array(heap_topk(cnt, 1, k))


def train(df, aids, ts, ops):
    topks = {}
    # for two modes
    for mode in [OP_WEIGHT, TIME_WEIGHT]:
        # get nested counter
        cnts = nb.typed.Dict.empty(
            key_type=nb.types.int64,
            value_type=nb.typeof(nb.typed.Dict.empty(key_type=nb.types.int64, value_type=nb.types.float64)))
        max_idx = len(df)
        for idx in tqdm(range(0, max_idx, PARALLEL)):
            row = df.iloc[idx:min(idx + PARALLEL, max_idx)
                          ][['session', 'idx', 'length', 'start_time']].values
            get_pairs(aids, ts, ops, row, cnts, OPS_WEIGTHS, mode)

        # get topk from counter
        topk = nb.typed.Dict.empty(
            key_type=nb.types.int64,
            value_type=nb.types.int64[:])
        get_topk(cnts, topk, TOPN)

        del cnts
        gc.collect()
        topks[mode] = topk


if __name__ == '__main__':
    df, aids, ts, ops = load_data('../data/raw/', 'test')
    train(df, aids, ts, ops)
