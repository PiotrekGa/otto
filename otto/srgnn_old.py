import os.path as osp

import math
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

from gensim.models import Word2Vec

import torch
from torch_geometric.data import Data, InMemoryDataset, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn.conv import MessagePassing
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard.summary import hparams
from torch.utils.tensorboard import SummaryWriter


timestamp = str(datetime.now())[:19].replace(
    '-', '').replace(':', '').replace(' ', '_')


class CONFIG:

    debug = True

    # tensorboard
    log_dir = f'runs/experiment{timestamp}'
    comment = ''

    # dataset
    data_path = '.'
    ignore_idx = 1855607
    dataset_size = None
    use_events = True
    w2v_path = 'raw/word2vec.model'
    w2v_size = 100

    # model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 128
    hidden_dim = 102
    epochs = 20
    l2_penalty = 0.00001
    weight_decay = 0.1
    step = 3
    lr = 0.001
    num_items = 1855608

    # validation
    valid_sessions = 10_000

    if debug:
        data_path = 'data/'
        w2v_path = 'data/word2vec.model'
        dataset_size = 2000
        batch_size = 32
        epochs = 3
        hidden_dim = 16
        valid_sessions = 1000


class PatchedSummaryWriter(SummaryWriter):
    def add_hparams(self, hparam_dict, metric_dict):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError(
                'hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        logdir = self._get_file_writer().get_logdir()

        with SummaryWriter(log_dir=logdir) as w_hp:
            w_hp.file_writer.add_summary(exp)
            w_hp.file_writer.add_summary(ssi)
            w_hp.file_writer.add_summary(sei)
            for k, v in metric_dict.items():
                w_hp.add_scalar(k, v)


class GraphInMemoryDataset(InMemoryDataset):
    def __init__(self, root, file_name, ignore_idx=1855607, use_events=True, use_subsessions=True, dataset_size=None, w2v_path=None, transform=None, pre_transform=None):
        self.file_name = file_name
        self.ignore_idx = ignore_idx
        self.use_events = use_events
        self.use_subsessions = use_subsessions
        self.dataset_size = dataset_size
        self.w2v_path = w2v_path
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f'{self.file_name}.parquet', f'{self.file_name}_labels.jsonl']

    @property
    def processed_file_names(self):
        if self.w2v_path:
            return [f'{self.file_name}_w2v.pt']
        else:
            return [f'{self.file_name}.pt']

    def download(self):
        pass

    @staticmethod
    def merge_two_lists(l1, l2):
        result = [None]*(len(l1)+len(l2))
        result[::2] = l1
        result[1::2] = l2
        return result

    @staticmethod
    def create_sessions(df, threshold=30):
        df['ts_lagged'] = df.ts.shift(1).fillna(df.ts).astype(np.int32)
        df['session_lagged'] = df.session.shift(
            1).fillna(df.ts).astype(np.int32)
        df['difference'] = df.ts - df.ts_lagged
        df['break_point'] = ((df.difference > threshold * 60)
                             | (df.session != df.session_lagged)) * 2 - 1
        df.drop(['ts_lagged', 'session_lagged',
                'difference'], axis=1, inplace=True)
        return df

    def process(self):
        raw_data_file1 = f'{self.raw_dir}/{self.raw_file_names[0]}'
        raw_data_file2 = f'{self.raw_dir}/{self.raw_file_names[1]}'
        sessions = pd.read_parquet(raw_data_file1)
        labels = pd.read_json(raw_data_file2, lines=True).set_index(
            'session')['labels'].iloc[:self.dataset_size]

        sessions = self.create_sessions(sessions)

        sessions_aids = sessions.groupby('session')['aid'].apply(list)
        if self.use_events:
            sessions.type = sessions.type + 1855603
            sessions_type = sessions.groupby('session')['type'].apply(list)
        if self.use_subsessions:
            sessions.break_point = sessions.break_point * 1855606
            sessions_subs = sessions.groupby(
                'session')['break_point'].apply(list)

        del sessions

        labels_df = pd.DataFrame(list(labels), index=labels.index)

        clicks_series = labels_df.clicks.dropna().astype(np.int32).reset_index()
        clicks_series.columns = ['session', 'aid']
        clicks_series['event_type'] = 'clicks'

        carts_series = labels_df.carts.explode().dropna().astype(np.int32).reset_index()
        carts_series.columns = ['session', 'aid']
        carts_series['event_type'] = 'carts'

        orders_series = labels_df.orders.explode().dropna().astype(np.int32).reset_index()
        orders_series.columns = ['session', 'aid']
        orders_series['event_type'] = 'orders'

        series_with_events = pd.concat([clicks_series, orders_series, carts_series]).sample(
            frac=1)

        series_with_events['rr'] = series_with_events.groupby(
            'event_type').cumcount() % int(series_with_events.shape[0] / 32)
        series_with_events.sort_values(
            by='rr', inplace=True, ignore_index=True)
        series_with_events.drop('rr', axis=1, inplace=True)

        del labels_df, clicks_series, orders_series, carts_series

        if self.w2v_path is not None:
            w2v_embedder = W2VEmbedding(self.w2v_path)

        data_list = []
        for _, row in tqdm(series_with_events.iterrows(), total=series_with_events.shape[0]):
            idx = row['session']

            first_list = self.merge_two_lists(
                sessions_type[idx], sessions_aids[idx])
            second_list = self.merge_two_lists(
                sessions_subs[idx], [-1] * len(sessions_subs[idx]))
            long_list = self.merge_two_lists(second_list, first_list)
            long_list = [i for i in long_list if i >= 0]

            if row['event_type'] == 'clicks':
                long_list = long_list + [1855603]
                session, y_clicks, y_carts, y_orders = long_list, row[
                    'aid'], self.ignore_idx, self.ignore_idx
            elif row['event_type'] == 'carts':
                long_list = long_list + [1855604]
                session, y_clicks, y_carts, y_orders = long_list, self.ignore_idx, row[
                    'aid'], self.ignore_idx
            elif row['event_type'] == 'orders':
                long_list = long_list + [1855605]
                session, y_clicks, y_carts, y_orders = long_list, self.ignore_idx, self.ignore_idx, row[
                    'aid']

            codes, uniques = pd.factorize(session)
            edge_index = np.array([codes[:-1], codes[1:]], dtype=np.int32)
            # making edge_index unique is not needed as MessagePassing does it with weights addded
            edge_index = torch.tensor(
                edge_index, dtype=torch.long)

            if self.w2v_path is not None:
                x = np.concatenate([w2v_embedder(i).reshape(1, -1)
                                   for i in uniques], axis=0)
                x = torch.tensor(x, dtype=torch.float32)
            else:
                x = torch.tensor(uniques, dtype=torch.long).unsqueeze(1)

            y_clicks = torch.tensor([y_clicks], dtype=torch.long)
            y_carts = torch.tensor([y_carts], dtype=torch.long)
            y_orders = torch.tensor([y_orders], dtype=torch.long)
            data_list.append(
                Data(x=x, edge_index=edge_index, y_clicks=y_clicks, y_carts=y_carts, y_orders=y_orders).contiguous())

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class GraphDataset(Dataset):

    def __init__(self, root, file_name, ignore_idx=1855607, use_events=True, use_subsessions=True, dataset_size=None, w2v_path=None, transform=None, pre_transform=None):
        self.file_name = file_name
        self.ignore_idx = ignore_idx
        self.use_events = use_events
        self.use_subsessions = use_subsessions
        self.dataset_size = dataset_size
        self.w2v_path = w2v_path

        if file_name == 'valid1__test':
            self.sessions_num = 2224034
        elif file_name == 'valid2__test':
            self.sessions_num = 1783737

        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return [f'{self.file_name}.parquet', f'{self.file_name}_labels.jsonl']

    @property
    def processed_file_names(self):
        if self.w2v_path:
            return [f'{self.file_name}_w2v_{i}.pt' for i in range(self.sessions_num)]
        else:
            return [f'{self.file_name}_{i}.pt' for i in range(self.sessions_num)]

    def download(self):
        pass

    @staticmethod
    def merge_two_lists(l1, l2):
        result = [None]*(len(l1)+len(l2))
        result[::2] = l1
        result[1::2] = l2
        return result

    @staticmethod
    def create_sessions(df, threshold=30):
        df['ts_lagged'] = df.ts.shift(1).fillna(df.ts).astype(np.int32)
        df['session_lagged'] = df.session.shift(
            1).fillna(df.ts).astype(np.int32)
        df['difference'] = df.ts - df.ts_lagged
        df['break_point'] = ((df.difference > threshold * 60)
                             | (df.session != df.session_lagged)) * 2 - 1
        df.drop(['ts_lagged', 'session_lagged',
                'difference'], axis=1, inplace=True)
        return df

    def process(self):
        raw_data_file1 = f'{self.raw_dir}/{self.raw_file_names[0]}'
        raw_data_file2 = f'{self.raw_dir}/{self.raw_file_names[1]}'
        sessions = pd.read_parquet(raw_data_file1)
        labels = pd.read_json(raw_data_file2, lines=True).set_index(
            'session')['labels'].iloc[:self.dataset_size]

        sessions = self.create_sessions(sessions)

        sessions_aids = sessions.groupby('session')['aid'].apply(list)
        if self.use_events:
            sessions.type = sessions.type + 1855603
            sessions_type = sessions.groupby('session')['type'].apply(list)
        if self.use_subsessions:
            sessions.break_point = sessions.break_point * 1855606
            sessions_subs = sessions.groupby(
                'session')['break_point'].apply(list)

        del sessions

        labels_df = pd.DataFrame(list(labels), index=labels.index)

        clicks_series = labels_df.clicks.dropna().astype(np.int32).reset_index()
        clicks_series.columns = ['session', 'aid']
        clicks_series['event_type'] = 'clicks'

        carts_series = labels_df.carts.explode().dropna().astype(np.int32).reset_index()
        carts_series.columns = ['session', 'aid']
        carts_series['event_type'] = 'carts'

        orders_series = labels_df.orders.explode().dropna().astype(np.int32).reset_index()
        orders_series.columns = ['session', 'aid']
        orders_series['event_type'] = 'orders'

        series_with_events = pd.concat([clicks_series, orders_series, carts_series]).sample(
            frac=1)

        series_with_events['rr'] = series_with_events.groupby(
            'event_type').cumcount() % int(series_with_events.shape[0] / 32)
        series_with_events.sort_values(
            by='rr', inplace=True, ignore_index=True)
        series_with_events.drop('rr', axis=1, inplace=True)

        del labels_df, clicks_series, orders_series, carts_series

        if self.w2v_path is not None:
            w2v_embedder = W2VEmbedding(self.w2v_path)

        for _, row in tqdm(series_with_events.iterrows(), total=series_with_events.shape[0]):
            idx = row['session']

            first_list = self.merge_two_lists(
                sessions_type[idx], sessions_aids[idx])
            second_list = self.merge_two_lists(
                sessions_subs[idx], [-1] * len(sessions_subs[idx]))
            long_list = self.merge_two_lists(second_list, first_list)
            long_list = [i for i in long_list if i >= 0]

            if row['event_type'] == 'clicks':
                long_list = long_list + [1855603]
                session, y_clicks, y_carts, y_orders = long_list, row[
                    'aid'], self.ignore_idx, self.ignore_idx
            elif row['event_type'] == 'carts':
                long_list = long_list + [1855604]
                session, y_clicks, y_carts, y_orders = long_list, self.ignore_idx, row[
                    'aid'], self.ignore_idx
            elif row['event_type'] == 'orders':
                long_list = long_list + [1855605]
                session, y_clicks, y_carts, y_orders = long_list, self.ignore_idx, self.ignore_idx, row[
                    'aid']

            codes, uniques = pd.factorize(session)
            edge_index = np.array([codes[:-1], codes[1:]], dtype=np.int32)
            # making edge_index unique is not needed as MessagePassing does it with weights added
            edge_index = torch.tensor(
                edge_index, dtype=torch.long)

            if self.w2v_path is not None:
                x = np.concatenate([w2v_embedder(i).reshape(1, -1)
                                   for i in uniques], axis=0)
                x = torch.tensor(x, dtype=torch.float32)
            else:
                x = torch.tensor(uniques, dtype=torch.long).unsqueeze(1)

            y_clicks = torch.tensor([y_clicks], dtype=torch.long)
            y_carts = torch.tensor([y_carts], dtype=torch.long)
            y_orders = torch.tensor([y_orders], dtype=torch.long)
            data = Data(x=x, edge_index=edge_index, y_clicks=y_clicks,
                        y_carts=y_carts, y_orders=y_orders).contiguous()

            if self.w2v_path:
                torch.save(data, osp.join(
                    self.processed_dir, f'{self.file_name}_w2v_{idx}.pt'))
            else:
                torch.save(data, osp.join(
                    self.processed_dir, f'{self.file_name}_{idx}.pt'))

    def len(self):
        return self.sessions_num

    def get(self, idx):
        if self.w2v_path:
            return torch.load(osp.join(self.processed_dir,
                                       f'{self.file_name}_w2v_{idx}.pt'))
        else:
            return torch.load(osp.join(self.processed_dir,
                                       f'{self.file_name}_{idx}.pt'))


class GraphInMemoryDatasetTest(InMemoryDataset):
    def __init__(self, root, file_name, use_events=True, use_subsessions=True, dataset_size=None, w2v_path=None, transform=None, pre_transform=None):
        self.file_name = file_name
        self.use_events = use_events
        self.use_subsessions = use_subsessions
        self.dataset_size = dataset_size
        self.w2v_path = w2v_path
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f'{self.file_name}.parquet']

    @property
    def processed_file_names(self):
        return [f'inference__{self.file_name}.pt']

    def download(self):
        pass

    @staticmethod
    def merge_two_lists(l1, l2):
        result = [None]*(len(l1)+len(l2))
        result[::2] = l1
        result[1::2] = l2
        return result

    @staticmethod
    def create_sessions(df, threshold=30):
        df['ts_lagged'] = df.ts.shift(1).fillna(df.ts).astype(np.int32)
        df['session_lagged'] = df.session.shift(
            1).fillna(df.ts).astype(np.int32)
        df['difference'] = df.ts - df.ts_lagged
        df['break_point'] = ((df.difference > threshold * 60)
                             | (df.session != df.session_lagged)) * 2 - 1
        df.drop(['ts_lagged', 'session_lagged',
                'difference'], axis=1, inplace=True)
        return df

    def process(self):
        raw_data_file1 = f'{self.raw_dir}/{self.raw_file_names[0]}'
        sessions = pd.read_parquet(raw_data_file1).iloc[:self.dataset_size, :]

        sessions = self.create_sessions(sessions)

        sessions_aids = sessions.groupby('session')['aid'].apply(list)
        if self.use_events:
            sessions.type = sessions.type + 1855603
            sessions_type = sessions.groupby('session')['type'].apply(list)
        if self.use_subsessions:
            sessions.break_point = sessions.break_point * 1855606
            sessions_subs = sessions.groupby(
                'session')['break_point'].apply(list)

        del sessions

        if self.w2v_path is not None:
            w2v_embedder = W2VEmbedding(self.w2v_path)

        data_list = []
        for idx in tqdm(sessions_aids.index):

            first_list = self.merge_two_lists(
                sessions_type[idx], sessions_aids[idx])
            second_list = self.merge_two_lists(
                sessions_subs[idx], [-1] * len(sessions_subs[idx]))
            long_list = self.merge_two_lists(second_list, first_list)
            long_list = [i for i in long_list if i >= 0]

            for event_type in [1855603, 1855604, 1855605]:
                session = long_list + [event_type]
                codes, uniques = pd.factorize(session)
                edge_index = np.array([codes[:-1], codes[1:]], dtype=np.int32)
                # making edge_index unique is not needed as MessagePassing does it with weights addded
                edge_index = torch.tensor(
                    edge_index, dtype=torch.long)
                if self.w2v_path is not None:
                    x = np.concatenate([w2v_embedder(i).reshape(1, -1)
                                        for i in uniques], axis=0)
                    x = torch.tensor(x, dtype=torch.float32)
                else:
                    x = torch.tensor(uniques, dtype=torch.long).unsqueeze(1)
                event_type = event_type - 1855603
                event_type = torch.tensor(event_type, dtype=torch.int8)
                session_id = torch.tensor(idx, dtype=torch.int32)
                data_list.append(
                    Data(x=x, edge_index=edge_index, event_type=event_type, session_id=session_id).contiguous())

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class GraphDatasetTest(Dataset):

    """
    file_size for test: 5015409
    file_size for valid2__test: 5351211
    """

    def __init__(self, root, file_name, file_size=5015409, use_events=True, use_subsessions=True, dataset_size=None, w2v_path=None, transform=None, pre_transform=None):
        self.file_name = file_name
        self.use_events = use_events
        self.file_size = file_size
        self.use_subsessions = use_subsessions
        self.dataset_size = dataset_size
        self.w2v_path = w2v_path
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return [f'{self.file_name}.parquet']

    @property
    def processed_file_names(self):
        return [f'inference__{self.file_name}_{i}.pt' for i in range(self.file_size)]

    def download(self):
        pass

    @staticmethod
    def merge_two_lists(l1, l2):
        result = [None]*(len(l1)+len(l2))
        result[::2] = l1
        result[1::2] = l2
        return result

    @staticmethod
    def create_sessions(df, threshold=30):
        df['ts_lagged'] = df.ts.shift(1).fillna(df.ts).astype(np.int32)
        df['session_lagged'] = df.session.shift(
            1).fillna(df.ts).astype(np.int32)
        df['difference'] = df.ts - df.ts_lagged
        df['break_point'] = ((df.difference > threshold * 60)
                             | (df.session != df.session_lagged)) * 2 - 1
        df.drop(['ts_lagged', 'session_lagged',
                'difference'], axis=1, inplace=True)
        return df

    def process(self):
        raw_data_file1 = f'{self.raw_dir}/{self.raw_file_names[0]}'
        sessions = pd.read_parquet(raw_data_file1).iloc[:self.dataset_size, :]

        sessions = self.create_sessions(sessions)

        sessions_aids = sessions.groupby('session')['aid'].apply(list)
        if self.use_events:
            sessions.type = sessions.type + 1855603
            sessions_type = sessions.groupby('session')['type'].apply(list)
        if self.use_subsessions:
            sessions.break_point = sessions.break_point * 1855606
            sessions_subs = sessions.groupby(
                'session')['break_point'].apply(list)

        del sessions

        if self.w2v_path is not None:
            w2v_embedder = W2VEmbedding(self.w2v_path)

        file_num = 0
        for idx in tqdm(sessions_aids.index):

            first_list = self.merge_two_lists(
                sessions_type[idx], sessions_aids[idx])
            second_list = self.merge_two_lists(
                sessions_subs[idx], [-1] * len(sessions_subs[idx]))
            long_list = self.merge_two_lists(second_list, first_list)
            long_list = [i for i in long_list if i >= 0]

            for event_type in [1855603, 1855604, 1855605]:
                session = long_list + [event_type]
                codes, uniques = pd.factorize(session)
                edge_index = np.array([codes[:-1], codes[1:]], dtype=np.int32)
                # making edge_index unique is not needed as MessagePassing does it with weights addded
                edge_index = torch.tensor(
                    edge_index, dtype=torch.long)
                if self.w2v_path is not None:
                    x = np.concatenate([w2v_embedder(i).reshape(1, -1)
                                        for i in uniques], axis=0)
                    x = torch.tensor(x, dtype=torch.float32)
                else:
                    x = torch.tensor(uniques, dtype=torch.long).unsqueeze(1)
                event_type = event_type - 1855603
                event_type = torch.tensor(event_type, dtype=torch.int8)
                session_id = torch.tensor(idx, dtype=torch.int32)
                data = Data(x=x, edge_index=edge_index,
                            event_type=event_type, session_id=session_id).contiguous()
                torch.save(data, osp.join(
                    self.processed_dir, f'inference__{self.file_name}_{file_num}.pt'))
                file_num += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir,
                          f'inference__{self.file_name}_{idx}.pt'))
        return data


class W2VEmbedding():
    def __init__(self, model_path):
        self.model_path = model_path
        self.w2v_model = Word2Vec.load(model_path)

        emb_1855603 = np.zeros(102, dtype=np.float32)
        emb_1855604 = np.concatenate([np.zeros(
            self.w2v_model.vector_size + 1, dtype=np.float32), np.ones(1, dtype=np.float32)])
        emb_1855605 = np.concatenate([np.zeros(self.w2v_model.vector_size, dtype=np.float32), np.ones(
            1, dtype=np.float32), np.zeros(1, dtype=np.float32)])
        emb_1855606 = np.concatenate([np.zeros(
            self.w2v_model.vector_size, dtype=np.float32), np.ones(2, dtype=np.float32)])

        self.types_embs = {'1855603': emb_1855603,
                           '1855604': emb_1855604,
                           '1855605': emb_1855605,
                           '1855606': emb_1855606}

    def get_embedding(self, x, types_embs, w2v_model):
        emb = types_embs.get(x)
        if emb is None:
            emb = np.concatenate(
                [w2v_model.wv.get_vector(x), np.zeros(2, dtype=np.float32)])
        return emb

    def __call__(self, x):
        emb = self.types_embs.get(str(x))
        if emb is None:
            emb = np.concatenate([self.w2v_model.wv.get_vector(
                str(x)), np.zeros(2, dtype=np.float32)])
        return emb


class GatedSessionGraphConv(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr: str = 'add', **kwargs):
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.gru = torch.nn.GRUCell(in_channels, out_channels, bias=False)

    def forward(self, x, edge_index):
        m = self.propagate(edge_index, x=x, size=None)
        x = self.gru(m, x)
        return x

    def message(self, x_j):
        return x_j

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)


class SRGNN(nn.Module):
    def __init__(self, hidden_size, n_items, w2v_size=None, w2v_path=None):
        super(SRGNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_items = n_items
        self.use_adapter = False
        if w2v_size is None:
            self.w2v = False
            self.embedding = nn.Embedding(self.n_items, self.hidden_size)
            self.output_size = hidden_size
        else:
            self.w2v_size = w2v_size
            self.w2v = True
            self.adapter = nn.Linear(self.w2v_size, self.hidden_size)
            self.output_size = w2v_size
            if w2v_size != self.hidden_size:
                self.use_adapter = True

            w2v_model = Word2Vec.load(w2v_path)
            vec = torch.tensor(np.vstack([w2v_model.wv.get_vector(str(i))
                                          for i in tqdm(range(len(w2v_model.wv)))]), dtype=torch.float32)
            self.w2v_vectors = torch.concat(
                [vec, torch.zeros((vec.shape[0], 2), dtype=torch.float32)], dim=1).to(CONFIG.device)

        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.gated = GatedSessionGraphConv(
            self.hidden_size, self.hidden_size)

        self.q = nn.Linear(self.hidden_size, 1)
        self.W_1 = nn.Linear(self.hidden_size,
                             self.hidden_size, bias=False)
        self.W_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_3 = nn.Linear(2 * self.hidden_size,
                             self.hidden_size, bias=False)

        self.W_clicks = nn.Linear(self.hidden_size,
                                  self.output_size, bias=False)
        self.W_carts = nn.Linear(self.hidden_size,
                                 self.output_size, bias=False)
        self.W_orders = nn.Linear(self.hidden_size,
                                  self.output_size, bias=False)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, data):
        x, edge_index, batch_map = data.x, data.edge_index, data.batch

        # (0)
        if self.use_adapter:
            embedding = torch.sigmoid(self.adapter(x).squeeze())
            embedding = self.batch_norm(embedding)
        elif self.w2v:
            embedding = x.squeeze()
        else:
            embedding = self.embedding(x).squeeze()
        # (1)-(5)
        v_i = self.gated(embedding, edge_index)

        # Divide nodes by session
        # For the detailed explanation of what is happening below, please refer
        # to the Medium blog post.
        sections = list(torch.bincount(batch_map).cpu())
        v_i_split = torch.split(v_i, sections)

        v_n, v_n_repeat = [], []
        for session in v_i_split:
            v_n.append(session[-1])
            v_n_repeat.append(
                session[-1].view(1, -1).repeat(session.shape[0], 1))
        v_n, v_n_repeat = torch.stack(v_n), torch.cat(v_n_repeat, dim=0)

        q1 = self.W_1(v_n_repeat)
        q2 = self.W_2(v_i)

        # (6)
        alpha = self.q(torch.sigmoid(q1 + q2))
        s_g_split = torch.split(alpha * v_i, sections)

        s_g = []
        for session in s_g_split:
            s_g_session = torch.sum(session, dim=0)
            s_g.append(s_g_session)
        s_g = torch.stack(s_g)

        # (7)
        s_l = v_n
        s_h = self.W_3(torch.cat([s_l, s_g], dim=-1))

        x1 = self.W_clicks(torch.sigmoid(s_h))
        x2 = self.W_carts(torch.sigmoid(s_h))
        x3 = self.W_orders(torch.sigmoid(s_h))

        # (8)

        if self.w2v:
            z1 = torch.mm(self.w2v_vectors, x1.T).T
            z2 = torch.mm(self.w2v_vectors, x2.T).T
            z3 = torch.mm(self.w2v_vectors, x3.T).T
        else:
            z1 = torch.mm(self.embedding.weight, x1.T).T
            z2 = torch.mm(self.embedding.weight, x2.T).T
            z3 = torch.mm(self.embedding.weight, x3.T).T

        return z1, z2, z3


def train(config):
    # Prepare data pipeline
    train_dataset = GraphInMemoryDataset(
        config.data_path, 'valid2__test', ignore_idx=config.ignore_idx, use_events=config.use_events, dataset_size=config.dataset_size, w2v_path=config.w2v_path)
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=False,
                              drop_last=True)
    val_dataset = GraphInMemoryDataset(
        config.data_path, 'valid1__test', ignore_idx=config.ignore_idx, use_events=config.use_events, dataset_size=config.valid_sessions, w2v_path=config.w2v_path)
    val_loader = DataLoader(val_dataset,
                            batch_size=config.batch_size,
                            shuffle=False,
                            drop_last=True)

    # Build model
    if config.w2v_size:
        input_vector_size = config.w2v_size + 2
    else:
        input_vector_size = None
    model = SRGNN(config.hidden_dim,
                  config.num_items, w2v_size=input_vector_size, w2v_path=config.w2v_path).to(config.device)

    # Get training components
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.lr,
                                 weight_decay=config.l2_penalty)
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=config.step,
                                          gamma=config.weight_decay)
    ignore_index = config.num_items - 1
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    writer = PatchedSummaryWriter(
        log_dir=config.log_dir, comment=config.comment)

    writer.add_hparams(
        {'batch_size': config.batch_size,
         'hidden_dim': config.hidden_dim,
         'epochs': config.epochs,
         'l2_penalty': config.l2_penalty,
         'weight_decay': config.weight_decay,
         'step': config.step,
         'lr': config.lr,
         'num_items': config.num_items},
        {})

    for epoch in range(config.epochs):
        total_loss = 0
        model.train()
        for _, batch in enumerate(tqdm(train_loader)):
            batch.to(config.device)
            optimizer.zero_grad()

            pred = model(batch)
            label_clicks = batch.y_clicks
            label_carts = batch.y_carts
            label_orders = batch.y_orders

            loss = criterion(pred[0], label_clicks) + \
                criterion(pred[1], label_carts) + \
                criterion(pred[2], label_orders)

            loss.backward()

            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        torch.save(model.state_dict(),
                   f'checkpoints/checkpoint_otto_{epoch}.pt')

        total_loss /= len(train_loader.dataset)

        writer.add_scalar('Loss/train', total_loss, epoch + 1)
        writer.add_scalar('Params/lr', optimizer.state_dict()
                          ['param_groups'][0]['lr'], epoch + 1)

        scheduler.step()

        test_acc_clicks, test_acc_carts, test_acc_orders, loss_test = test(
            val_loader, model)
        print(test_acc_clicks, test_acc_carts, test_acc_orders)

        writer.add_scalar('Accuracy/test_acc_clicks',
                          test_acc_clicks, epoch + 1)
        writer.add_scalar('Accuracy/test_acc_carts', test_acc_carts, epoch + 1)
        writer.add_scalar('Accuracy/test_acc_orders',
                          test_acc_orders, epoch + 1)
        writer.add_scalar('Loss/test', loss_test, epoch + 1)

    writer.close()

    return model


def test(loader, test_model, config=CONFIG):
    test_model.eval()

    correct_clicks = 0
    correct_carts = 0
    correct_orders = 0
    total_clicks = 0
    total_carts = 0
    total_orders = 0

    ignore_index = config.num_items - 1
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    loss_test_total = 0

    for _, data in enumerate(tqdm(loader)):
        data.to(config.device)
        with torch.no_grad():
            # max(dim=1) returns values, indices tuple; only need indices
            score = test_model(data)
            pred_clicks = score[0].max(dim=1)[1]
            pred_carts = score[1].max(dim=1)[1]
            pred_orders = score[2].max(dim=1)[1]
            label_clicks = data.y_clicks
            label_carts = data.y_carts
            label_orders = data.y_orders

            loss_test = criterion(score[0], label_clicks) + \
                criterion(score[1], label_carts) + \
                criterion(score[2], label_orders)

            loss_test_total += loss_test.item() * data.num_graphs

        correct_clicks += pred_clicks.eq(
            label_clicks)[label_clicks != config.ignore_idx].sum().item()
        total_clicks += sum(sum([label_clicks != config.ignore_idx])).item()

        correct_carts += pred_carts.eq(
            label_carts)[label_carts != config.ignore_idx].sum().item()
        total_carts += sum(sum([label_carts != config.ignore_idx])).item()

        correct_orders += pred_orders.eq(
            label_orders)[label_orders != config.ignore_idx].sum().item()
        total_orders += sum(sum([label_orders != config.ignore_idx])).item()

    loss_test_total /= len(loader.dataset)

    return correct_clicks / total_clicks, correct_carts / total_carts, correct_orders / total_orders, loss_test_total


def prepare_kaggle_submission(model, test_loader, sub_name, k_init=25, k_final=20, debug=False):
    submission = []
    for _, data in enumerate(tqdm(test_loader)):
        data.to(CONFIG.device)
        with torch.no_grad():
            # max(dim=1) returns values, indices tuple; only need indices
            score = model(data)

            score_click = torch.topk(score[0], k_init)[1]
            score_click = score_click.cpu().detach().numpy()

            score_cart = torch.topk(score[1], k_init)[1]
            score_cart = score_cart.cpu().detach().numpy()

            score_order = torch.topk(score[2], k_init)[1]
            score_order = score_order.cpu().detach().numpy()

            event_types = data.event_type
            sessions = data.session_id

        for row in range(event_types.size(0)):

            event_type = event_types[row].item()
            sessions_str = str(sessions[row].item()) + '_'

            if event_type == 0:
                top_k_pred = score_click[row]
                top_k_pred = top_k_pred[top_k_pred < 1855603][:k_final]
                sessions_str = sessions_str + 'clicks'

            elif event_type == 1:
                top_k_pred = score_cart[row]
                top_k_pred = top_k_pred[top_k_pred < 1855603][:k_final]
                sessions_str = sessions_str + 'carts'

            else:
                top_k_pred = score_order[row]
                top_k_pred = top_k_pred[top_k_pred < 1855603][:k_final]
                sessions_str = sessions_str + 'orders'

            top_k_pred = ' '.join(list(top_k_pred.astype(str)))
            submission.append([sessions_str, top_k_pred])
        torch.cuda.empty_cache()
        if debug:
            if len(submission) > 1000:
                break

    submission = pd.DataFrame(submission, columns=['session_type', 'labels'])
    submission.to_csv(f'{sub_name}.csv', index=False)


if __name__ == '__main__':
    model = train(CONFIG)
