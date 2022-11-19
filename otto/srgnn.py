import copy

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch_geometric.data import Data, Dataset, DataLoader, InMemoryDataset
from torch_geometric.nn.conv import MessagePassing
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data


class CONFIG:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 100
    hidden_dim = 32
    epochs = 100
    l2_penalty = 0.00001
    weight_decay = 0.1
    step = 30
    lr = 0.001
    num_items = 1855603


class GraphDataset(InMemoryDataset):
    def __init__(self, root, file_name, transform=None, pre_transform=None):
        self.file_name = file_name
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

    def process(self):
        raw_data_file1 = f'{self.raw_dir}/{self.raw_file_names[0]}'
        raw_data_file2 = f'{self.raw_dir}/{self.raw_file_names[1]}'
        sessions = pd.read_parquet(raw_data_file1).iloc[:1000, :]
        labels = pd.read_json(raw_data_file2, lines=True).set_index(
            'session')['labels']

        sessions = sessions.loc[sessions.type == 0, ['session', 'aid']]
        ses_len = sessions.session.value_counts()
        ses_len = ses_len[ses_len > 1]
        sessions['len'] = sessions.session.map(ses_len)
        del ses_len
        sessions.dropna(inplace=True)
        sessions = sessions.groupby('session')['aid'].apply(list)
        data_list = []

        for idx in tqdm(sessions.index):
            session, y = sessions[idx], labels[idx].get('clicks', None)
            if y:
                codes, uniques = pd.factorize(session)
                edge_index = np.array([codes[:-1], codes[1:]], dtype=np.int32)
                edge_index = torch.tensor(edge_index, dtype=torch.long)
                x = torch.tensor(uniques, dtype=torch.long).unsqueeze(1)
                y = torch.tensor([y], dtype=torch.long)
                data_list.append(Data(x=x, edge_index=edge_index, y=y))

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
        alpha = self.q(F.sigmoid(q1 + q2))
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
    train_dataset = GraphDataset('../data', 'train')
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=False,
                              drop_last=True)
    val_dataset = GraphDataset('../data', 'test')
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
    criterion = nn.CrossEntropyLoss()

    # Train
    losses = []
    test_accs = []
    top_k_accs = []

    best_acc = 0
    best_model = None

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
            score = test_model(data)
            pred = score.max(dim=1)[1]
            label = data.y

        if save_model_preds:
            data = {}
            data['pred'] = pred.view(-1).cpu().detach().numpy()
            data['label'] = label.view(-1).cpu().detach().numpy()

            df = pd.DataFrame(data=data)
            # Save locally as csv
            df.to_csv('pred.csv', sep=',', index=False)

        correct += pred.eq(label).sum().item()

        # We calculate Hit@K accuracy only at test time.
        if not is_validation:
            score = score.cpu().detach().numpy()
            for row in range(pred.size(0)):
                top_k_pred = np.argpartition(score[row], -k)[-k:]
                if label[row].item() in top_k_pred:
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
