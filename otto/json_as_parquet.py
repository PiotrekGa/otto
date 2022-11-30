import pandas as pd
import numpy as np

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


def main():
    files = ['valid0__test']
    for file in files:
        print(f'processing {file}')
        path_in = f'../data/raw/{file}.jsonl'
        path_out = f'../data/raw/{file}.parquet'
        df = jsonl_to_df(path_in)
        df.to_parquet(path_out, index=False)


if __name__ == '__main__':
    main()
