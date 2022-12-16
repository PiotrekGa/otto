import pandas as pd
from tqdm import tqdm
from gensim.models import Word2Vec


class CONFIG:
    use_types = True
    min_count = 5


if __name__ == '__main__':

    for window in tqdm([35]):
        df1 = pd.read_parquet('../data/raw/test.parquet')
        df2 = pd.read_parquet('../data/raw/train.parquet')
        sessions = pd.concat([df1, df2])
        del df1, df2

        if CONFIG.use_types:
            sessions.aid = sessions.aid.astype(
                str) + '_' + sessions.type.astype(str)
        else:
            sessions.aid = sessions.aid.astype(str)

        aid_cnt = sessions.aid.value_counts()
        aid_cnt = aid_cnt[aid_cnt < CONFIG.min_count]
        aid_cnt = pd.Series(aid_cnt.index.str.slice(-1), index=aid_cnt.index)
        sessions.aid = sessions.aid.map(
            aid_cnt).fillna(sessions.aid).astype(str)

        sessions_aids = sessions.groupby('session')['aid'].apply(list)

        x = [' '.join(sessions_aids[idx])
             for idx in tqdm(sessions_aids.index)]
        with open('w2v_input.txt', 'w') as f:
            for line in x:
                f.write(f"{line}\n")

        model = Word2Vec(corpus_file=f"w2v_input.txt",
                         vector_size=100, window=window, min_count=CONFIG.min_count, workers=8)

        window = str(window).zfill(2)
        model_path = f'../data/raw/word2vec_w{window}.model'
        model.save(model_path)
