import polars as pl
import numpy as np
import pandas as pd
import gc
from time import sleep
from tqdm import tqdm


import logging
from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import GRU4Rec
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger

from typing import List
import torch

from pydantic import BaseModel
from recbole.data import create_dataset
from recbole.data.interaction import Interaction
from recbole.utils import get_model, init_seed


class CONFIG:
    fold = 'valid2__'
    days_back_train = 7


MAX_ITEM = 30

parameter_dict = {
    'data_path': '',
    'USER_ID_FIELD': 'session',
    'ITEM_ID_FIELD': 'aid',
    'TIME_FIELD': 'ts',
    'user_inter_num_interval': "[5,Inf)",
    'item_inter_num_interval': "[5,Inf)",
    'load_col': {'inter': ['session', 'aid', 'ts']},
    'train_neg_sample_args': None,
    'epochs': 10,
    'stopping_step': 3,
    'num_layers': 1,
    'embedding_size': 512,
    'hidden_size': 512,
    'eval_batch_size': 1024,
    'train_batch_size': 1024,
    'MAX_ITEM_LIST_LENGTH': MAX_ITEM,
    'eval_args': {
        'split': {'RS': [9, 1, 0]},
        'group_by': 'user',
        'order': 'TO',
        'mode': 'full'}
}

recbole_config = Config(
    model='GRU4Rec', dataset='recbox_data', config_dict=parameter_dict)

# init random seed
init_seed(recbole_config['seed'], recbole_config['reproducibility'])

# logger initialization
init_logger(recbole_config)
logger = getLogger()

# Create handlers
c_handler = logging.StreamHandler()
c_handler.setLevel(logging.INFO)
logger.addHandler(c_handler)

# write config info into log
logger.info(recbole_config)


def create_atiomic_file(config):
    train = pl.read_parquet(f'raw/{config.fold}train.parquet')
    max_ts = train.select(pl.col('ts')).max()[0, 0]
    train = train.filter(pl.col('ts') > max_ts -
                         config.days_back_train * 24 * 3600)

    cnt_ses = train.groupby('session').agg(
        pl.col('ts').count().alias('ses_cnt'))
    cnt_ses = cnt_ses.filter(pl.col('ses_cnt') >= 5).drop('ses_cnt')
    train = train.join(cnt_ses, on='session')

    sessions = train.select('session').unique()
    sessions = sessions.sample(frac=0.2)
    train = train.join(sessions, on='session')

    del cnt_ses, sessions

    test = pl.read_parquet(f'raw/{config.fold}test.parquet')

    df = pl.concat([train, test])

    df = df.sort(['session', 'aid', 'ts'])
    df = df.with_columns((pl.col('ts') * 1e9).alias('ts'))
    df = df.rename({'session': 'session:token',
                   'aid': 'aid:token', 'ts': 'ts:float'})

    df['session:token', 'aid:token', 'ts:float'].write_csv(
        'recbox_data/recbox_data.inter', sep='\t')

    del df, train, test
    gc.collect()


def train(config_recbole):
    dataset = create_dataset(config_recbole)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(
        config_recbole, dataset)

    model = GRU4Rec(config_recbole, train_data.dataset).to(
        config_recbole['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = Trainer(config_recbole, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
    #best_valid_score, best_valid_result = trainer.fit(train_data)

    del trainer, train_data, valid_data, test_data
    gc.collect()
    return model, dataset


class ItemHistory(BaseModel):
    sequence: List[str]
    topk: int


class RecommendedItems(BaseModel):
    score_list: List[float]
    item_list: List[str]


def pred_user_to_item(dataset, model, item_history: ItemHistory):
    item_history_dict = item_history.dict()
    item_sequence = item_history_dict["sequence"]
    item_length = len(item_sequence)
    pad_length = MAX_ITEM  # pre-defined by recbole

    padded_item_sequence = torch.nn.functional.pad(
        torch.tensor(dataset.token2id(dataset.iid_field, item_sequence)),
        (0, pad_length - item_length),
        "constant",
        0,
    )

    input_interaction = Interaction(
        {
            "aid_list": padded_item_sequence.reshape(1, -1),
            "item_length": torch.tensor([item_length]),
        }
    )
    scores = model.full_sort_predict(input_interaction.to(model.device))
    scores = scores.view(-1, dataset.item_num)
    scores[:, 0] = -np.inf  # pad item score -> -inf
    topk_score, topk_iid_list = torch.topk(scores, item_history_dict["topk"])

    predicted_score_list = topk_score.tolist()[0]
    predicted_item_list = dataset.id2token(
        dataset.iid_field, topk_iid_list.tolist()
    ).tolist()

    recommended_items = {
        "score_list": predicted_score_list,
        "item_list": predicted_item_list,
    }
    return recommended_items


def create_reco(dataset, model, config):
    test = pd.read_parquet(f'raw/{config.fold}test.parquet')
    test.drop_duplicates(subset=['session', 'aid'], keep='last', inplace=True)
    session_types = ['clicks']
    test_session_AIDs = test.groupby('session')['aid'].apply(list)

    del test
    gc.collect()

    labels = []

    cnt_fail = 0
    cnt = 0

    avail_tokens = set(dataset.field2token_id['aid'].keys())

    test_session_AIDs = test_session_AIDs

    type_weight_multipliers = {0: 1, 1: 6, 2: 3}
    for AIDs in tqdm(test_session_AIDs, total=len(test_session_AIDs)):
        if cnt % 10000 == 0 and cnt > 0:
            print(cnt_fail / cnt)
        AIDs = list(dict.fromkeys(AIDs))
        AIDs = [str(i) for i in AIDs if str(i) in avail_tokens]
        if len(AIDs) > 0:
            AIDs = AIDs[-MAX_ITEM:]
            item = ItemHistory(sequence=AIDs, topk=MAX_ITEM)
            AIDs = []
            try:
                nns = [int(v) for v in pred_user_to_item(
                    dataset, model, item)['item_list']]
            except:
                nns = []
                cnt_fail += 1
                AIDs = []

            for word in nns:
                AIDs.append(word)
        else:
            cnt_fail += 1
        cnt += 1

        labels.append(AIDs)

    print('FAILED:', cnt_fail / len(test_session_AIDs))

    labels_as_strings = [' '.join([str(l) for l in lls]) for lls in labels]
    predictions = pd.DataFrame(
        data={'session_type': test_session_AIDs.index, 'labels': labels_as_strings})

    prediction_dfs = []

    for st in session_types:
        modified_predictions = predictions.copy()
        modified_predictions.session_type = modified_predictions.session_type.astype(
            'str') + f'_{st}'
        prediction_dfs.append(modified_predictions)

    submission = pd.concat(prediction_dfs).reset_index(drop=True)
    submission.to_csv(f'{config.fold}recbole.csv', index=False)


if __name__ == '__main__':
    sleep(1)
    create_atiomic_file(CONFIG)
    model, dataset = train(recbole_config)
    create_reco(dataset, model, CONFIG)
