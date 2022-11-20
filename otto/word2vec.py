import pandas as pd
from tqdm import tqdm
from gensim.models import Word2Vec

if __name__ == '__main__':
    df1 = pd.read_parquet('../data/raw/test.parquet')
    df2 = pd.read_parquet('../data/raw/train.parquet')
    sessions = pd.concat([df1, df2])
    del df1, df2
    sessions.type = sessions.type + 1855603
    sessions_aids = sessions.groupby('session')['aid'].apply(list)
    sessions_type = sessions.groupby('session')['type'].apply(list)

    def merge_two_lists(l1, l2):
        result = [None]*(len(l1)+len(l2))
        result[::2] = l1
        result[1::2] = l2
        return [str(i) for i in result]

    x = [' '.join(merge_two_lists(sessions_type[idx], sessions_aids[idx]))
         for idx in tqdm(sessions_aids.index)]
    with open('w2v_input.txt', 'w') as f:
        for line in x:
            f.write(f"{line}\n")

    model = Word2Vec(corpus_file="w2v_input.txt",
                     vector_size=100, window=5, min_count=1, workers=4)
    model.save("word2vec.model")
