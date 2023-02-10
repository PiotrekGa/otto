import os
import polars as pl
from scipy import sparse
import numpy as np
import pandas as pd
import joblib
import pickle
from pathlib import Path
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import StandardScaler, LabelEncoder, normalize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class CONFIG:
    emb_size = 1024
    n_iter = 5
    n_sketches = 40
    n_sketches_random = 40
    sketch_dim = 128


DATAPATH = ''
FOLD = ''


class CleoraCovisit():

    def __init__(self, fold, emb_size, emb_name, n_iter, days_back, session_hist, before_time, after_time, left_types, right_types, type_weight, time_weight_coef) -> None:
        self.maxs = {'': 1662328791,
                     'valid3__': 1661723998,
                     'valid2__': 1661119195,
                     'valid1__': 1660514389}
        self.fold = fold
        self.emb_size = emb_size
        self.emb_name = emb_name
        self.n_iter = n_iter
        self.days_back = days_back
        self.session_hist = session_hist
        self.before_time = before_time * 3600
        self.after_time = after_time * 3600
        self.left_types = left_types
        self.right_types = right_types
        self.type_weight = type_weight
        self.time_weight_coef = time_weight_coef

    def compute_embedding(self):
        max_ts = self.maxs[self.fold]
        min_ts = max_ts - (24 * 60 * 60 * self.days_back)

        df = pl.read_parquet(
            f'{DATAPATH}raw/{self.fold}test.parquet')
        df1 = pl.read_parquet(
            f'{DATAPATH}raw/{self.fold}train.parquet')
        df = pl.concat([df, df1])
        del df1

        df = df.filter(pl.col('ts') >= min_ts)

        df = df.sort(by=['session', 'ts'], reverse=[False, True])
        df = df.with_column(
            pl.col('session').cumcount().over('session').alias('n'))
        df = df.filter(pl.col('n') < self.session_hist).drop('n')
        df = df.join(df, on='session')
        df = df.filter(((pl.col('ts_right') - pl.col('ts')) >= - self.before_time) & ((pl.col(
            'ts_right') - pl.col('ts')) <= self.after_time) & (pl.col('aid') != pl.col('aid_right')))
        df = df.filter(pl.col('type').is_in(self.left_types) &
                       pl.col('type_right').is_in(self.right_types))

        aid_left_cnts = df.groupby('aid').count()
        aid_left_cnts = aid_left_cnts.filter(
            pl.col('count') >= 5).drop('count')
        df = df.join(aid_left_cnts, on='aid')
        aid_left_cnts.columns = ['aid_right']
        df = df.join(aid_left_cnts, on='aid_right')

        df = df.with_column(pl.col('type_right').apply(
            lambda x: self.type_weight[x]).alias('wgt'))
        df = df.with_column(pl.col('wgt') * (1 + self.time_weight_coef *
                            ((pl.col('ts') - min_ts) / (max_ts - min_ts))))
        df = df.select(['aid', 'aid_right', 'wgt'])
        df = df.groupby(['aid', 'aid_right']).agg(pl.col('wgt').sum())
        df = df.sort(by=['aid', 'wgt'], reverse=[False, True])
        # df = df.with_column(pl.col('aid').cumcount().over('aid').alias('n'))
        aid_wgt_sum = df.groupby('aid').agg(
            pl.col('wgt').sum().alias('wgt_sum'))
        df = df.join(aid_wgt_sum, on='aid')
        df = df.with_column(
            pl.col('wgt') / pl.col('wgt_sum')).drop('wgt_sum')

        aid_left_cnts = aid_left_cnts.to_numpy().ravel()
        le = LabelEncoder()
        le.fit(aid_left_cnts)
        df = df.to_pandas()

        df.aid = le.transform(df.aid)
        df.aid_right = le.transform(df.aid_right)

        df.aid = df.aid.astype(np.int32)
        df.aid_right = df.aid_right.astype(np.int32)

        emb = np.random.rand(df.aid.max() + 1, self.emb_size) * 2 - 1
        x = sparse.coo_matrix((df.wgt, (df.aid, df.aid_right)),
                              shape=(df.aid.max() + 1, df.aid.max() + 1))
        x = x.tocsr()
        emb = sparse.csr_matrix(emb)
        for _ in range(self.n_iter):
            emb = x.dot(emb)
            emb = np.array(emb.todense())
            emb = normalize(emb, norm='l2', axis=1)
            emb = sparse.csr_matrix(emb)
        emb = np.array(emb.todense())
        emb = pd.DataFrame(emb)
        emb.columns = [
            f'emb_{self.emb_name}_{str(i)}' for i in range(self.emb_size)]
        emb.to_csv(
            f'{DATAPATH}cleora/{self.fold}emb_{self.emb_name}.csv', index=False)
        joblib.dump(
            le, f'{DATAPATH}cleora/{self.fold}cleora_lables_{self.emb_name}.pkl')

    def load_embedding(self):
        embedding_file = Path(
            f'{DATAPATH}cleora/{self.fold}emb_{self.emb_name}.csv')
        if not embedding_file.is_file():
            print(f'preparing embeddings {self.fold}{self.emb_name}')
            self.compute_embedding()
        return pd.read_csv(embedding_file.as_posix()).values

    def load_le(self):
        return joblib.load(f'{DATAPATH}cleora/{self.fold}cleora_lables_{self.emb_name}.pkl')


class Coder(object):
    def __init__(self, n_sketches, sketch_dim):
        self.n_sketches = n_sketches
        self.sketch_dim = sketch_dim
        self.ss = StandardScaler()
        self.sp = GaussianRandomProjection(n_components=16*n_sketches)

    def fit(self, v):
        self.ss = self.ss.fit(v)
        vv = self.ss.transform(v)
        self.sp = self.sp.fit(vv)
        vvv = self.sp.transform(vv)
        self.init_biases(vvv)

    def transform(self, v):
        v = self.ss.transform(v)
        v = self.sp.transform(v)
        v = self.discretize(v)
        v = np.packbits(v, axis=-1)
        v = np.frombuffer(np.ascontiguousarray(v), dtype=np.uint16).reshape(
            v.shape[0], -1) % self.sketch_dim
        return v

    def transform_to_absolute_codes(self, v, labels=None):
        codes = self.transform(v)
        pos_index = np.array(
            [i*self.sketch_dim for i in range(self.n_sketches)], dtype=np.int_)
        index = codes + pos_index
        return index


class DLSH(Coder):
    def __init__(self, n_sketches, sketch_dim):
        super().__init__(n_sketches, sketch_dim)

    def init_biases(self, v):
        self.biases = np.array(
            [np.percentile(v[:, i], q=50, axis=0) for i in range(v.shape[1])])

    def discretize(self, v):
        return ((np.sign(v - self.biases)+1)/2).astype(np.uint8)


def compute_emde(fold, emb_size, n_iter, n_sketches, sketch_dim, n_sketches_random):

    print('load emb')
    modalities = []
    cc1 = CleoraCovisit(fold=fold, emb_size=emb_size, emb_name='cleora1', n_iter=n_iter, days_back=7, session_hist=30, before_time=0,
                        after_time=2, left_types=[0, 1, 2], right_types=[0, 1, 2], type_weight={0: 1, 1: 6, 2: 3}, time_weight_coef=3)
    cc1_emb = cc1.load_embedding()

    codes = compute_codes(cc1_emb, n_sketches, sketch_dim)
    modalities.append(codes)

    print('create random emb')
    random_embeddings = np.random.normal(0, 0.1, size=[len(cc1_emb), 1024])
    random_codes = compute_codes(
        random_embeddings, n_sketches=n_sketches_random, sketch_dim=sketch_dim)
    modalities.append(random_codes)

    aids = cc1.load_le()
    aids = [str(i) for i in aids.classes_]

    print('merging modealities')
    aid2codes = merge_modalities(aids, modalities, offsets=[
        i * n_sketches * sketch_dim for i in range(len(modalities))])

    with open(os.path.join(f'{DATAPATH}cleora/', f'{fold}_aid2codes'), 'wb') as handle:
        pickle.dump(aid2codes, handle, protocol=pickle.HIGHEST_PROTOCOL)


def compute_codes(embeddings: np.ndarray, n_sketches: int, sketch_dim: int):
    """
    Compute LSH codes
    """
    vcoder = DLSH(n_sketches, sketch_dim)
    vcoder.fit(embeddings)
    codes = vcoder.transform_to_absolute_codes(embeddings)
    return codes


def merge_modalities(aids, modalities, offsets):
    aid2codes = {}
    for j, aid in enumerate(aids):
        codes = []
        for i, modality in enumerate(modalities):
            codes.append(modality[j] + offsets[i])
        aid2codes[str(aid)] = list(np.concatenate(codes))
    return aid2codes


class Model(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim):
        super().__init__()

        self.output_dim = output_dim

        self.l1 = nn.Linear(input_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, hidden_size)
        self.l_output = nn.Linear(hidden_size, self.output_dim)
        self.projection = nn.Linear(input_dim, hidden_size)

        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.bn4 = nn.BatchNorm1d(hidden_size)

    def forward(self, sketches):
        """
        Feed forward network with residual connections.
        """
        x_proj = self.projection(sketches)
        x_ = self.bn1(F.leaky_relu(self.l1(sketches)))
        x = self.bn2(F.leaky_relu(self.l2(x_) + x_proj))
        x = self.bn3(F.leaky_relu(self.l3(x) + x_proj))
        x = self.l_output(self.bn4(F.leaky_relu(self.l4(x) + x_)))
        x = F.softmax(x.view(-1, self.output_dim))
        return x


def codes_to_sketch(codes, input_dim):
    """
    Convert abosulte codes into sketch sparse vector
    """
    x = np.zeros(input_dim)
    for ind in codes:
        x[ind] += 1
    return x


def sketch_session(df, aid2codes, input_dim, sketch_dim, le, le2, decay_value=.9):
    sketches = []
    targets = []

    for sess in tqdm(df):
        targets.append(le2.transform([sess[-1]]).astype(np.int32))
        sess = sess[:-1]
        codes = aid2codes[str(sess[0])]
        sketch = codes_to_sketch(codes, input_dim)
        if len(sess) > 1:
            for ses in sess[1:]:
                sketch *= decay_value
                sketch += codes_to_sketch(aid2codes[str(ses)], input_dim)
        sketch = normalize(sketch.reshape(-1, sketch_dim),
                           'l2').reshape((input_dim, 1))
        sketch = sketch.astype(np.float32)
        sketches.append(sketch)

    sketches = np.hstack(sketches).T
    sketches = sketches.reshape(len(y), -1, 1)
    sketches = torch.tensor(sketches, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.long)

    return sketches, targets


def create_dataset(fold, input_dim, sketch_dim):
    with open(os.path.join(f'{DATAPATH}cleora/', f'{fold}_aid2codes'), 'rb') as f:
        aid2codes = pickle.load(f)
    df = pd.read_parquet(f'{DATAPATH}raw/{fold}test.parquet')
    le = joblib.load(f'{DATAPATH}cleora/{fold}cleora_lables_cleora1.pkl')
    aids_uq = set([int(i) for i in aid2codes.keys()])
    df = df.loc[df.aid.isin(aids_uq), :]

    last = df.drop_duplicates(subset=['session'], keep='last')
    last_aids = last.aid.value_counts()
    last_aids = last_aids[last_aids >= 10]
    last = last.loc[last.aid.isin(last_aids.index), 'session']
    df = df.loc[df.session.isin(last)]

    le2 = LabelEncoder()
    le2.fit(last_aids.index)
    joblib.dump(le2, f'{DATAPATH}cleora/{fold}le2.pkl')

    del last, last_aids
    df = df.groupby('session')['aid'].apply(list)
    df = df.loc[df.apply(len) > 1]
    df = df.sample(frac=1.)

    split_idx = int(df.shape[0] * 0.9)
    df_train = df.iloc[:split_idx]
    df_valid = df.iloc[split_idx:]

    print('creating train dataset')
    train_sketches, train_targets = sketch_session(
        df_train, aid2codes, input_dim, sketch_dim, le, le2)
    print('creating valid dataset')
    valid_sketches, valid_targets = sketch_session(
        df_valid, aid2codes, input_dim, sketch_dim, le, le2)

    train_dataset = TensorDataset(train_sketches, train_targets)
    valid_dataset = TensorDataset(valid_sketches, valid_targets)

    return train_dataset, valid_dataset


def main(config):
    compute_emde(FOLD, config.emb_size, config.n_iter, n_sketches=config.n_sketches,
                 sketch_dim=config.sketch_dim, n_sketches_random=config.n_sketches_random)
    input_dim = (config.n_sketches + config.n_sketches_random) * \
        config.sketch_dim
    train_dataset, valid_dataset = create_dataset(
        FOLD, input_dim, config.sketch_dim)
    torch.save(train_dataset, f'{FOLD}train_dataset.pt')
    torch.save(valid_dataset, f'{FOLD}valid_dataset.pt')


if __name__ == '__main__':
    main(CONFIG)
