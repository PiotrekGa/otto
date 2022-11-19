from torch.utils.tensorboard.summary import hparams
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import time
import joblib
import os


import torch
from torch_geometric.data import Data, InMemoryDataset, DataListLoader
from torch_geometric.utils import undirected
from torch_geometric.nn.models import Node2Vec

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import optuna

timestamp = str(datetime.now())[:19].replace(
    '-', '').replace(':', '').replace(' ', '_')


class CONFIG:

    debug = False

    task = 'compute'
    dataset = 'valid__'
    min_cnt = 100

    # tensorboard
    log_dir = f'runs/experiment{timestamp}'
    comment = ''

    # node2vec
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sc_name = 'CyclicLR_triangular2'
    sc_cooldown = 1
    sc_factor = 0.1
    batch_size = 1024
    lr = 0.001
    epochs = 5
    embedding_dim = 30
    walk_length = 20
    context_size = 5
    walks_per_node = 10
    num_negative_samples = 5
    p = 2
    q = 1
    sparse = False


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


class N2VDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 name,
                 min_cnt=1,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        self.name = name
        self.min_cnt = min_cnt
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f'{self.name}train.parquet', f'{self.name}test.parquet']

    @property
    def processed_file_names(self):
        return [f'{self.name}n2v_user_base.pt']

    def download(self):
        pass

    def process(self):

        df = pd.read_parquet(f'{self.raw_dir}/{self.name}test.parquet')
        df = df.loc[:, ['session', 'aid']].drop_duplicates()

        if not CONFIG.debug:
            train = pd.read_parquet(f'{self.raw_dir}/{self.name}train.parquet')
            train = train.loc[:, ['session', 'aid']].drop_duplicates()
            df = pd.concat([train, df])
            del train

        aids_cnt = df.groupby('aid').count()['session']
        aids_cnt = aids_cnt[aids_cnt >= self.min_cnt]

        print('********', df.shape)
        df = df.loc[df.aid.isin(aids_cnt.index), :]
        print('********', df.shape)

        del aids_cnt

        aid_map = {aid: i for i, aid in enumerate(set(list(df.aid.unique())))}
        to_add = max(aid_map.values()) + 1
        ses_map = {ses: i + to_add for i,
                   ses in enumerate(set(list(df.session.unique())))}
        aid_map_inv = {aid_map[i]: i for i in aid_map.keys()}
        ses_map_inv = {ses_map[i]: i for i in ses_map.keys()}
        mapping_inv = {**aid_map_inv, **ses_map_inv}

        df.session = df.session.map(ses_map)
        df.aid = df.aid.map(aid_map)

        df = df.values.astype(np.int32)
        df = np.vstack([df, df[:, [1, 0]]])

        edge_index = torch.tensor(df.T, dtype=torch.long, device=CONFIG.device)
        del df

        aids = set(aid_map.values())
        y = [(i in aids) * 1 for i in list(mapping_inv.keys())]
        y = torch.tensor(y, dtype=torch.int8, device=CONFIG.device)

        mapping_inv = pd.Series(
            mapping_inv).reset_index().values.astype(np.int32)
        mapping_inv = torch.tensor(
            mapping_inv, dtype=torch.int32, device=CONFIG.device)

        data = Data(edge_index=edge_index, y=y,
                    num_nodes=len(y), mapping_inv=mapping_inv).contiguous()

        data_list = [data]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def compute_n2v(data, config):

    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in tqdm(loader, total=len(loader)):
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(config.device),
                              neg_rw.to(config.device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def test():
        x = model().cpu().numpy()
        y = data['y'].cpu().numpy()

        np.random.seed(42)
        idx = np.random.choice(len(y), size=10_000, replace=False)
        x = x[idx, :]
        y = y[idx]

        del idx

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, shuffle=True, random_state=42, stratify=y)
        del x, y
        m = LogisticRegression()
        m.fit(x_train, y_train)
        y_test_hat = m.predict_proba(x_test)[:, 1]
        return roc_auc_score(y_test, y_test_hat)

    writer = PatchedSummaryWriter(
        log_dir=config.log_dir, comment=config.comment)

    torch.manual_seed(42)
    model = Node2Vec(data.edge_index, embedding_dim=config.embedding_dim, walk_length=config.walk_length,
                     context_size=config.context_size, walks_per_node=config.walks_per_node,
                     num_negative_samples=config.num_negative_samples, p=config.p, q=config.q, sparse=config.sparse).to(config.device)

    loader = model.loader(batch_size=config.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=config.lr)
    if config.sc_name == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=config.sc_patience, cooldown=config.sc_cooldown, factor=config.sc_factor, verbose=True)
    elif config.sc_name == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10, eta_min=0)
    elif config.sc_name == 'CyclicLR_triangular2':
        optimizer = torch.optim.RMSprop(
            model.parameters(), lr=config.lr, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=config.lr / 10, max_lr=config.lr, step_size_up=1, step_size_down=5, mode="triangular2")
    elif config.sc_name == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=config.lr / 10)

    writer.add_hparams(
        {'batch_size': config.batch_size,
         'lr': config.lr,
         'walk_length': config.walk_length,
         'context_size': config.context_size,
         'walks_per_node': config.walks_per_node,
         'num_negative_samples': config.num_negative_samples,
         'p': config.p,
         'q': config.q},
        {})

    for epoch in range(config.epochs):
        print(f'EPOCH: {epoch + 1}')
        torch.cuda.empty_cache()
        loss = train()
        acc = test()
        scheduler.step()
        torch.save(model.state_dict(),
                   f'checkpoints/checkpoint_otto_{epoch}.pt')
        writer.add_scalar('Loss/train', loss, epoch + 1)
        writer.add_scalar('Accuracy/test', acc, epoch + 1)
        writer.add_scalar('Params/lr', optimizer.state_dict()
                          ['param_groups'][0]['lr'], epoch + 1)

    writer.add_hparams(
        {'batch_size': config.batch_size,
         'lr': config.lr,
         'walk_length': config.walk_length,
         'context_size': config.context_size,
         'walks_per_node': config.walks_per_node,
         'num_negative_samples': config.num_negative_samples,
         'p': config.p,
         'q': config.q},
        {'hparam/accuracy': acc,
         'hparam/loss': loss})

    @torch.no_grad()
    def get_emb(model):
        model.eval()
        z = model().cpu().numpy()
        return z

    if config.task == 'compute':
        emb = get_emb(model)
        emb = pd.DataFrame(emb)
        writer.close()
        return emb, acc
    elif config.task == 'optimize':
        return acc


def objective(trial):

    joblib.dump(study, 'study.pkl')

    timestamp = str(datetime.now())[:19].replace(
        '-', '').replace(':', '').replace(' ', '_')

    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    sc_name = trial.suggest_categorical('sc_name', [
                                        'CosineAnnealingWarmRestarts', 'CyclicLR_triangular2', 'ReduceLROnPlateau', 'CosineAnnealingLR'])
    batch_size = trial.suggest_int('batch_size', 16, 128)
    walk_length = trial.suggest_int('walk_length', 5, 6)
    context_size = trial.suggest_int('context_size', 2, 4)
    walks_per_node = trial.suggest_int('walks_per_node', 2, 3)
    num_negative_samples = trial.suggest_int('num_negative_samples', 1, 3)
    p = trial.suggest_float('p', 1e-4, 10, log=True)
    q = trial.suggest_float('q', 1e-4, 10, log=True)

    CONFIG.log_dir = f'runs_optuna/experiment{timestamp}'
    CONFIG.lr = lr
    CONFIG.sc_name = sc_name
    CONFIG.batch_size = batch_size
    CONFIG.walk_length = walk_length
    CONFIG.context_size = context_size
    CONFIG.walks_per_node = walks_per_node
    CONFIG.num_negative_samples = num_negative_samples
    CONFIG.p = p
    CONFIG.q = q

    try:
        score = compute_n2v(data, CONFIG)
    except:
        score = -1

    return score


if __name__ == '__main__':
    start = time.time()
    dataset = N2VDataset('.', CONFIG.dataset, CONFIG.min_cnt)
    data = dataset[0]
    if CONFIG.task == 'compute':
        emb, score = compute_n2v(data, CONFIG)
        emb.to_csv(f'{CONFIG.dataset}otto_emb.csv')
    elif CONFIG.task == 'optimize':
        if os.path.exists('study.pkl'):
            study = joblib.load('study.pkl')
        else:
            study = optuna.create_study(direction='maximize')

        # study.optimize(objective, timeout=3600 * 20)
        study.optimize(objective, n_trials=2)

        joblib.dump(study, 'study.pkl')

        print('best score:', study.best_value)
        print('best params:', study.best_params)
    else:
        print('UNKNOWN TASK')

    print(f'script executed in {round(time.time() - start)} seconds')
