import pandas as pd
import numpy as np
from tqdm import tqdm

ID2TYPE = ['clicks', 'carts', 'orders']
TYPE2ID = {a: i for i, a in enumerate(ID2TYPE)}


def jsonl_to_df(path):
    sessions = []
    aids = []
    tss = []
    types = []

    chunks = pd.read_json(path, lines=True, chunksize=100_000)

    for chunk in chunks:
        for _, session_data in chunk.iterrows():
            num_events = len(session_data.events)
            sessions += ([session_data.session] * num_events)
            for event in session_data.events:
                aids.append(event['aid'])
                tss.append(int(event['ts'] / 1000))
                types.append(TYPE2ID[event['type']])

    df = pd.DataFrame(data={'session': sessions, 'aid': aids, 'ts': tss, 'type': types},
                      dtype=np.int32)
    df['type'] = df['type'].astype(np.uint8)
    return df


def labels_to_parquet(file_name):
    labels = pd.read_json(
        f'../data/raw/{file_name}__test_labels.jsonl', lines=True)
    new_labels = []
    for _, row in tqdm(labels.iterrows(), total=labels.shape[0]):
        if 'clicks' in row.labels:
            label = row.labels['clicks']
            type = 0
            new_labels.append([row.session, type, label])
        if 'carts' in row.labels:
            for label in row.labels['carts']:
                type = 1
                new_labels.append([row.session, type, label])
        if 'orders' in row.labels:
            for label in row.labels['orders']:
                type = 2
                new_labels.append([row.session, type, label])
    new_labels = pd.DataFrame(new_labels, columns=['session', 'type', 'aid'])
    new_labels = new_labels.astype(np.int32)
    new_labels.type = new_labels.type.astype(np.int8)
    new_labels.to_parquet(f'../data/raw/{file_name}__test_labels.parquet')


def main():
    files = ['valid1__train', 'valid2__train', 'valid3__train']
    for file in tqdm(files):
        print(f'processing {file}')
        path_in = f'data/raw/{file}.jsonl'
        path_out = f'data/raw/{file}.parquet'
        df = jsonl_to_df(path_in)
        df.to_parquet(path_out, index=False)


if __name__ == '__main__':
    main()
