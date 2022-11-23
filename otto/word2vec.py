import pandas as pd
from tqdm import tqdm
from gensim.models import Word2Vec

if __name__ == '__main__':
    df1 = pd.read_parquet('../data/raw/test.parquet')
    df2 = pd.read_parquet('../data/raw/train.parquet')
    sessions = pd.concat([df1, df2])
    del df1, df2
    sessions.aid = sessions.aid.astype(str)
    sessions_aids = sessions.groupby('session')['aid'].apply(list)

    x = [' '.join(sessions_aids[idx])
         for idx in tqdm(sessions_aids.index)]
    with open('../data/w2v_input.txt', 'w') as f:
        for line in x:
            f.write(f"{line}\n")

    model = Word2Vec(corpus_file="../data/w2v_input.txt",
                     vector_size=100, window=5, min_count=1, workers=7)
    model.save("../data/word2vec.model")
