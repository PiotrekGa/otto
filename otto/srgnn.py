import copy
import os.path as osp

import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn.conv import MessagePassing
import torch.nn as nn
import torch.nn.functional as F
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
    ignore_idx = 1855606
    dataset_size = None
    use_events = True

    # model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 128
    hidden_dim = 32
    epochs = 50
    l2_penalty = 0.00001
    weight_decay = 0.1
    step = 15
    lr = 0.001
    num_items = 1855607

    if debug:
        dataset_size = 1000
        batch_size = 64
        epochs = 2


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
    def __init__(self, root, file_name, ignore_idx, use_events=True, dataset_size=None, transform=None, pre_transform=None):
        self.file_name = file_name
        self.ignore_idx = ignore_idx
        self.use_events = use_events
        self.dataset_size = dataset_size
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f'{self.file_name}.parquet', f'{self.file_name}_labels.jsonl']

    @property
    def processed_file_names(self):
        return [f'{self.file_name}.pt']

    def download(self):
        pass

    @staticmethod
    def merge_two_lists(l1, l2):
        result = [None]*(len(l1)+len(l2))
        result[::2] = l1
        result[1::2] = l2
        return result

    def process(self):
        raw_data_file1 = f'{self.raw_dir}/{self.raw_file_names[0]}'
        raw_data_file2 = f'{self.raw_dir}/{self.raw_file_names[1]}'
        sessions = pd.read_parquet(raw_data_file1)
        labels = pd.read_json(raw_data_file2, lines=True).set_index(
            'session')['labels'].iloc[:self.dataset_size]

        sessions_aids = sessions.groupby('session')['aid'].apply(list)
        if self.use_events:
            sessions.type = sessions.type + 1855603
            sessions_type = sessions.groupby('session')['type'].apply(list)
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
            frac=1).reset_index(drop=True)

        del labels_df, clicks_series, orders_series, carts_series

        data_list = []
        for _, row in tqdm(series_with_events.iterrows(), total=series_with_events.shape[0]):
            idx = row['session']
            if self.use_events:
                if row['event_type'] == 'clicks':
                    session, y_clicks, y_carts, y_orders = self.merge_two_lists(
                        sessions_type[idx], sessions_aids[idx]), row['aid'], self.ignore_idx, self.ignore_idx
                elif row['event_type'] == 'carts':
                    session, y_clicks, y_carts, y_orders = self.merge_two_lists(
                        sessions_type[idx], sessions_aids[idx]), self.ignore_idx, row['aid'], self.ignore_idx
                elif row['event_type'] == 'orders':
                    session, y_clicks, y_carts, y_orders = self.merge_two_lists(
                        sessions_type[idx], sessions_aids[idx]), self.ignore_idx, self.ignore_idx, row['aid']
            else:
                if row['event_type'] == 'clicks':
                    session, y_clicks, y_carts, y_orders = sessions_aids[
                        idx], row['aid'], self.ignore_idx, self.ignore_idx
                elif row['event_type'] == 'carts':
                    session, y_clicks, y_carts, y_orders = sessions_aids[
                        idx], self.ignore_idx, row['aid'], self.ignore_idx
                elif row['event_type'] == 'orders':
                    session, y_clicks, y_carts, y_orders = sessions_aids[
                        idx], self.ignore_idx, self.ignore_idx, row['aid']

            codes, uniques = pd.factorize(session)
            edge_index = np.array([codes[:-1], codes[1:]], dtype=np.int32)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            x = torch.tensor(uniques, dtype=torch.long).unsqueeze(1)
            y_clicks = torch.tensor([y_clicks], dtype=torch.long)
            y_carts = torch.tensor([y_carts], dtype=torch.long)
            y_orders = torch.tensor([y_orders], dtype=torch.long)
            data_list.append(
                Data(x=x, edge_index=edge_index, y_clicks=y_clicks, y_carts=y_carts, y_orders=y_orders))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class GatedSessionGraphConv(MessagePassing):
    def __init__(self, out_channels, aggr: str = 'add', **kwargs):
        super().__init__(aggr=aggr, **kwargs)

        self.out_channels = out_channels

        self.gru = torch.nn.GRUCell(out_channels, out_channels, bias=False)

    def forward(self, x, edge_index):
        m = self.propagate(edge_index, x=x, size=None)
        x = self.gru(m, x)
        return x

    def message(self, x_j):
        return x_j

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)


class SRGNN(nn.Module):
    def __init__(self, hidden_size, n_items):
        super(SRGNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_items = n_items

        self.embedding = nn.Embedding(self.n_items, self.hidden_size)
        self.gated = GatedSessionGraphConv(self.hidden_size)

        self.q = nn.Linear(self.hidden_size, 1)
        self.W_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_3 = nn.Linear(2 * self.hidden_size,
                             self.hidden_size, bias=False)

        self.W_clicks = nn.Linear(self.hidden_size,
                                  self.hidden_size, bias=False)
        self.W_carts = nn.Linear(self.hidden_size,
                                 self.hidden_size, bias=False)
        self.W_orders = nn.Linear(self.hidden_size,
                                  self.hidden_size, bias=False)

    def reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, data):
        x, edge_index, batch_map = data.x, data.edge_index, data.batch

        # (0)
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
        z1 = torch.mm(self.embedding.weight, x1.T).T
        z2 = torch.mm(self.embedding.weight, x2.T).T
        z3 = torch.mm(self.embedding.weight, x3.T).T

        return z1, z2, z3


def train(config):
    # Prepare data pipeline
    train_dataset = GraphInMemoryDataset(
        'data/', 'valid1__test', ignore_idx=config.ignore_idx, use_events=config.use_events, dataset_size=config.dataset_size)
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=False,
                              drop_last=True)
    val_dataset = GraphInMemoryDataset(
        'data/', 'valid2__test', ignore_idx=config.ignore_idx, use_events=config.use_events, dataset_size=config.dataset_size)
    val_loader = DataLoader(val_dataset,
                            batch_size=config.batch_size,
                            shuffle=False,
                            drop_last=True)

    # Build model
    model = SRGNN(config.hidden_dim, config.num_items).to(config.device)

    # Get training components
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.lr,
                                 weight_decay=config.l2_penalty)
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=config.step,
                                          gamma=config.weight_decay)
    ignore_index = config.num_items - 1
    criterion1 = nn.CrossEntropyLoss(ignore_index=ignore_index)
    criterion2 = nn.CrossEntropyLoss(ignore_index=ignore_index)
    criterion3 = nn.CrossEntropyLoss(ignore_index=ignore_index)

    # Train
    losses = []
    test_accs = []
    top_k_accs = []

    best_acc = 0
    best_model = None

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

            loss = criterion1(pred[0], label_clicks) + \
                criterion2(pred[1], label_carts) + \
                criterion3(pred[2], label_orders)

            loss.backward()
            print(loss)

            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

            writer.add_scalar('Loss/train', loss, epoch + 1)
            writer.add_scalar('Params/lr', optimizer.state_dict()
                              ['param_groups'][0]['lr'], epoch + 1)

        total_loss /= len(train_loader.dataset)
        losses.append(total_loss)

        scheduler.step()

        if epoch % 1 == 0:
            test_acc, top_k_acc = test(val_loader, model, is_validation=True)
            print(test_acc)
            test_accs.append(test_acc)
            top_k_accs.append(top_k_acc)
            if test_acc > best_acc:
                best_acc = test_acc
                best_model = copy.deepcopy(model)
        else:
            test_accs.append(test_accs[-1])

        writer.add_scalar('Accuracy/test_acc', test_acc, epoch + 1)
        writer.add_scalar('Accuracy/top_k_acc', top_k_acc, epoch + 1)

    writer.close()

    return test_accs, top_k_accs, losses, best_model, best_acc, val_loader


def test(loader, test_model, is_validation=False, save_model_preds=False, config=CONFIG):
    test_model.eval()

    # Define K for Hit@K metrics.
    k = 20
    correct = 0
    top_k_correct = 0

    for _, data in enumerate(tqdm(loader)):
        data.to(config.device)
        with torch.no_grad():
            # max(dim=1) returns values, indices tuple; only need indices
            score = test_model(data)[0]
            pred = score.max(dim=1)[1]
            label_clicks = data.y_clicks

        if save_model_preds:
            data = {}
            data['pred'] = pred.view(-1).cpu().detach().numpy()
            data['label'] = label_clicks.view(-1).cpu().detach().numpy()

            df = pd.DataFrame(data=data)
            # Save locally as csv
            df.to_csv('pred.csv', sep=',', index=False)

        correct += pred.eq(label_clicks).sum().item()

        # We calculate Hit@K accuracy only at test time.
        if not is_validation:
            score = score.cpu().detach().numpy()
            for row in range(pred.size(0)):
                top_k_pred = np.argpartition(score[row], -k)[-k:]
                if label_clicks[row].item() in top_k_pred:
                    top_k_correct += 1

    if not is_validation:
        return correct / len(loader), top_k_correct / len(loader)
    else:
        return correct / len(loader), 0


if __name__ == '__main__':
    test_accs, top_k_accs, losses, best_model, best_acc, test_loader = train(
        CONFIG)
    print(test_accs, top_k_accs)
    print("Maximum test set accuracy: {0}".format(max(test_accs)))
    print("Minimum loss: {0}".format(min(losses)))
