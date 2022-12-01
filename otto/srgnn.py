import math
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
from torch_geometric.data import Data, InMemoryDataset
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

    # dataset
    event_type = 'orders'
    train_set = 'valid2__test'
    valid_set = 'valid3__test'
    data_path = '.'
    dataset_size = None
    use_events = False
    use_subsessions = False

    # tensorboard
    log_dir = f'runs_{event_type}/experiment{timestamp}'
    comment = ''

    # model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 128
    hidden_dim = 128
    epochs = 20
    l2_penalty = 0.00001
    weight_decay = 0.1
    step = 3
    lr = 0.001
    num_items = 1855608

    # validation
    if event_type == 'clicks':
        valid_sessions = 10_000
    else:
        valid_sessions = 50_000

    # submission
    do_submission = False
    model_path = f'checkpoints/checkpoint_otto_{event_type}_{epochs-1}.pt'
    submission_name = 'sub2'
    submission_size = None

    if debug:
        data_path = 'data/'
        dataset_size = 100000
        batch_size = 32
        epochs = 5
        hidden_dim = 32
        valid_sessions = 5000
        submission_size = 1000
        model_path = f'checkpoints/checkpoint_otto_{event_type}_{epochs-1}.pt'


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
    def __init__(self, root, file_name, event_type, inference=False, dataset_size=None, transform=None, pre_transform=None):
        self.file_name = file_name
        self.event_type = event_type
        mapper = {'clicks': 0, 'carts': 1, 'orders': 2}
        self.event_code = mapper[event_type]
        self.inference = inference
        self.dataset_size = dataset_size
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f'{self.file_name}.parquet']

    @property
    def processed_file_names(self):
        if self.inference:
            return [f'inference_{self.file_name}_{self.event_type}.pt']
        else:
            return [f'{self.file_name}_{self.event_type}.pt']

    def download(self):
        pass

    def process(self):
        raw_data_file1 = f'{self.raw_dir}/{self.raw_file_names[0]}'
        sessions = pd.read_parquet(raw_data_file1).iloc[:self.dataset_size, :]

        types = sessions.groupby('session')['type'].apply(list)
        sessions = sessions.groupby('session')['aid'].apply(list)

        if self.inference:
            sessions = sessions.to_frame()
        else:
            aids = sessions.loc[sessions.apply(len) > 1]
            types = types.loc[types.apply(len) > 1]

            aids = aids.apply(lambda x: x[::-1])
            types = types.apply(lambda x: x[::-1])
            filter = types.apply(lambda x: self.event_code in x) * (types.apply(lambda x: self.event_code !=
                                                                                x[-1]) | types.apply(lambda x: sum([self.event_code == i for i in x]) > 1))

            aids = aids.loc[filter]
            types = types.loc[filter]
            type_idx = types.apply(lambda x: x.index(self.event_code))
            sessions = pd.concat([aids, types, type_idx], axis=1)
            sessions.columns = ['aid', 'types', 'idx']

        data_list = []
        for _, row in sessions.iterrows():
            if self.inference:
                seq = row.aid
            else:
                y = torch.tensor([row.aid[row.idx]], dtype=torch.long)
                seq = row.aid[row.idx + 1:][::-1]

            codes, uniques = pd.factorize(seq)
            edge_index = np.array([codes[:-1], codes[1:]], dtype=np.int32)
            edge_index = torch.tensor(edge_index, dtype=torch.long)

            x = torch.tensor(uniques, dtype=torch.long).unsqueeze(1)

            if self.inference:
                data_list.append(
                    Data(x=x, edge_index=edge_index).contiguous())
            else:
                data_list.append(
                    Data(x=x, edge_index=edge_index, y=y).contiguous())

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

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
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

        # (8)
        z = torch.mm(self.embedding.weight, s_h.T).T
        return z


def train(config):
    # Prepare data pipeline
    train_dataset = GraphInMemoryDataset(
        config.data_path, config.train_set, config.event_type, dataset_size=config.dataset_size)
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              drop_last=True)

    val_dataset = GraphInMemoryDataset(
        config.data_path, config.valid_set, config.event_type, dataset_size=config.valid_sessions)
    val_loader = DataLoader(val_dataset,
                            batch_size=config.batch_size,
                            shuffle=True,
                            drop_last=True)

    # Build model
    model = SRGNN(config.hidden_dim,
                  config.num_items).to(config.device)

    # Get training components
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.lr,
                                 weight_decay=config.l2_penalty)
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=config.step,
                                          gamma=config.weight_decay)
    criterion = nn.CrossEntropyLoss()

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
            label = batch.y

            loss = criterion(pred, label)

            loss.backward()

            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        torch.save(model.state_dict(),
                   f'checkpoints/checkpoint_otto_{config.event_type}_{epoch}.pt')

        total_loss /= len(train_loader.dataset)

        writer.add_scalar('Loss/train', total_loss, epoch + 1)
        writer.add_scalar('Params/lr', optimizer.state_dict()
                          ['param_groups'][0]['lr'], epoch + 1)

        scheduler.step()

        test_acc, loss_test, top_k_correct = test(
            val_loader, model)
        print('EPOCH', epoch, test_acc, loss_test, total_loss, top_k_correct)

        writer.add_scalar('Accuracy/test_acc',
                          test_acc, epoch + 1)
        writer.add_scalar('Accuracy/top_20_correct', top_k_correct, epoch + 1)
        writer.add_scalar('Loss/test', loss_test, epoch + 1)

    writer.close()

    return model


def test(loader, test_model, config=CONFIG):
    test_model.eval()

    correct = 0

    criterion = nn.CrossEntropyLoss()
    loss_test_total = 0
    top_k_correct = 0

    for _, data in enumerate(tqdm(loader)):
        data.to(config.device)
        with torch.no_grad():
            # max(dim=1) returns values, indices tuple; only need indices
            score = test_model(data)
            pred = score.max(dim=1)[1]
            label = data.y

            loss_test = criterion(score, label)

            loss_test_total += loss_test.item() * data.num_graphs

        correct += pred.eq(label).sum().item()

        for row in range(pred.size(0)):
            top_k_pred = np.argpartition(score[row].cpu(), -20)[-20:]
            if label[row].item() in top_k_pred:
                top_k_correct += 1

    loss_test_total /= len(loader.dataset)
    correct /= len(loader.dataset)
    top_k_correct /= len(loader.dataset)

    return correct, loss_test_total, top_k_correct


def prepare_kaggle_submission(config, k_init=25, k_final=20, debug=False):

    model = SRGNN(config.hidden_dim,
                  config.num_items).to(config.device)
    model.load_state_dict(torch.load(config.model_path))

    dataset = GraphInMemoryDataset(
        config.data_path, config.valid_set, use_events=config.use_events, use_subsessions=config.use_subsessions, dataset_size=config.submission_size)

    loader = DataLoader(dataset,
                        batch_size=config.batch_size,
                        shuffle=False,
                        drop_last=False)

    submission = []
    for _, data in enumerate(tqdm(loader)):
        data.to(CONFIG.device)
        with torch.no_grad():
            # max(dim=1) returns values, indices tuple; only need indices
            score = model(data)
            score = score.cpu().detach().numpy()
            sessions = data.session_id

        for row in range(score.shape[0]):
            sessions_str = str(sessions[row].item()) + '_'
            top_k_pred = np.argpartition(score[row], -k_init)[-k_init:]
            top_k_pred = top_k_pred[top_k_pred < 1855603][:k_final]
            sessions_str = sessions_str + config.event_type
            top_k_pred = ' '.join(list(top_k_pred.astype(str)))
            submission.append([sessions_str, top_k_pred])
        torch.cuda.empty_cache()
        if debug:
            if len(submission) > 1000:
                break

    submission = pd.DataFrame(submission, columns=['session_type', 'labels'])
    submission.to_csv(
        f'{config.submission_name}_{config.event_type}.csv', index=False)


if __name__ == '__main__':
    model = train(CONFIG)
    if CONFIG.do_submission:
        prepare_kaggle_submission(CONFIG)
