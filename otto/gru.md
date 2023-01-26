candidates.py 最下面

```
from recbole.quick_start import load_data_and_model
from typing import List, Tuple
from pydantic import BaseModel
import torch
from recbole.data.interaction import Interaction
import pandas as pd
import numpy as np

from collections import defaultdict
from recbole.config import Config
from recbole.utils import init_seed, init_logger
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.quick_start import load_data_and_model
from recbole.model.sequential_recommender import GRU4Rec


class ItemHistory(BaseModel):
    sequence: List[str]
    topk: int

class RecommendedItems(BaseModel):
    score_list: List[float]
    item_list: List[str]

class GRU(CandiadateGen):
    def __init__(self, fold, cfg,name, data_path, max_cands, type_weight=None ):
        super().__init__(fold=fold, name=name, data_path=data_path)

        self.fold = fold
        self.name = name
        self.cfg = cfg
        # self.event_type_str = event_type_str
        self.data_path = data_path
        self.max_cands = max_cands
        self.type_weight = type_weight

    def prepare_candidates(self):
        candidate_file = Path(f'{self.data_path}recbox/recbox.inter')
        if not candidate_file.is_file():
            print(f'生成inter')
            test = pl.read_parquet(f'{self.data_path}raw/test.parquet')
            valid2_test = pl.read_parquet(f'{self.data_path}raw/valid2__test.parquet')
            valid3_test = pl.read_parquet(f'{self.data_path}raw/valid3__test.parquet')

            df = pl.concat([valid2_test, valid3_test, test])
            del test, valid2_test, valid3_test

            # df = pl.read_parquet(f'{self.data_path}raw/{self.fold}test.parquet')
            df = df.sort(['session', 'aid', 'ts'])
            df = df.with_columns((pl.col('ts') * 1e9).alias('ts'))
            df = df.rename({'session': 'session:token', 'aid': 'aid:token', 'ts': 'ts:float'})
            if not os.path.exists(f'{self.data_path}recbox'):
                os.makedirs(f'{self.data_path}recbox')
            df['session:token', 'aid:token', 'ts:float',].write_csv(f'{self.data_path}recbox/recbox.inter', sep='\t')
            del df
        
        print('############ config')
        config = Config(model='GRU4Rec', dataset='recbox', config_dict=self.cfg.parameter_dict)
        init_seed(config['seed'], config['reproducibility'])

        model_file = Path(f'{self.data_path}/recbox/{self.cfg.gru_model_name}')
        if model_file.is_file():
            print('############ 加载模型和数据')
            _, model, dataset, train_data, valid_data, test_data = load_data_and_model(
                model_file=f'{self.data_path}/recbox/{self.cfg.gru_model_name}')
        else: 
            print('############ 模型不存在，从头开始')
            candidate_file = Path(f'{self.data_path}recbox/recbox-dataset.pth')
            if not candidate_file.is_file():
                print('############ 生成dataset')
                dataset = create_dataset(config)
                train_data, valid_data, test_data = data_preparation(config, dataset)
            else:
                print('############ 读取dataset')
                self.cfg.parameter_dict['dataset_save_path'] = f'{self.data_path}recbox/'
                self.cfg.parameter_dict['dataloaders_save_path'] = f'{self.data_path}recbox/'
                dataset = create_dataset(config)
                train_data, valid_data, test_data = data_preparation(config, dataset)
            
                # print('############ 读取model')
                # _, model, dataset, train_data, valid_data, test_data = load_data_and_model(
                #     model_file=f'{self.data_path}/recbox/{self.cfg.gru_model_name}',
                # )
            model = GRU4Rec(config, train_data.dataset).to(config['device'])
            trainer = Trainer(config, model)
            print('############ 开始训练model')
            best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
            print(f'best_valid_score: {best_valid_score}')
            print(f'best_valid_result: {best_valid_result}')



        
        # ======  Create recommendation result from trained model ======

        print('############ 训练完成，读取test')

        test_df = pl.read_parquet(f'{self.data_path}raw/{self.fold}test.parquet')
        # session_types = ['clicks', 'carts', 'orders']
        test_session_AIDs = test_df.to_pandas().reset_index(drop=True).groupby('session')['aid'].apply(list)
        test_session_types = test_df.to_pandas().reset_index(drop=True).groupby('session')['type'].apply(list)
        del test_df
        labels = []

        type_weight_multipliers = {0: 1, 1: 6, 2: 3}
        for AIDs, types in zip(test_session_AIDs, test_session_types):
            if len(AIDs) >= 20:
                # if we have enough aids (over equals 20) we don't need to look for candidates! we just use the old logic
                weights=np.logspace(0.1,1,len(AIDs),base=2, endpoint=True)-1  # logspac用于创建等比数列,开始点和结束点是10的幂
                aids_temp=defaultdict(lambda: 0)
                for aid,w,t in zip(AIDs,weights,types): 
                    aids_temp[aid]+= w * type_weight_multipliers[t]
                    
                sorted_aids=[k for k, v in sorted(aids_temp.items(), key=lambda item: -item[1])]  
                labels.append(sorted_aids[:20])
            else:
                AIDs = list(dict.fromkeys(AIDs))
                item = ItemHistory(sequence=AIDs, topk=self.max_cands)
                try:
                    nns = [ int(v) for v in self.pred_user_to_item(item, dataset, model, self.max_cands)['item_list']]
                except:
                    nns = []

                for word in nns:
                    if len(AIDs) == self.max_cands:
                        break
                    if int(word) not in AIDs:
                        AIDs.append(word)

                labels.append(AIDs[:self.max_cands]) 
        # labels_as_strings = [' '.join([str(l) for l in lls]) for lls in labels]
        # cands = pl.DataFrame(cands, orient='row', columns=['aid_str', 'aid', 'col_name', 'rank'])

        data = pd.DataFrame(data={'session': test_session_AIDs.index, 'aid': labels})        
        # df = pl.DataFrame(data)
        # print(f'--------- data:\n{data}')

        df = pl.DataFrame(data)
        # print(f'--------- df\n:{df}')
        df = df.explode('aid') 
        print(f'--------- after df\n:{df}')
        # cast 数据类型转换
        df = df.select(
            [pl.col('session').cast(pl.Int32), pl.col('aid').cast(pl.Int32)])
        df.write_parquet(
            f'{self.data_path}candidates/{self.fold}{self.name}.parquet')
        print(f'############ 保存成功{self.data_path}candidates/{self.fold}{self.name}.parquet')
        # df.to_csv(f'./data/raw/gru4rec.csv', index=False)
        # df = pl.read_csv(f'{self.data_path}raw/{self.fold}{self.base_file_name}.csv').lazy()       

    def pred_user_to_item(item_history: ItemHistory, dataset, model, max_cands):
        item_history_dict = item_history.dict()
        item_sequence = item_history_dict["sequence"]  # sequence is AIDs
        item_length = len(item_sequence)
        pad_length = max_cands  

        '''
        First, we need to use token2id() to convert external user id 
        into internal user id.
        Then, we create a 0 padded tensor to pass into the interaction object. 
        The number of 0s depends on the length of the original item list. 
        If there are 4 items, then its padded with 16 0s so that the total 
        length is 20, which is what we want to predict.
        '''
        # 不足 20 个候选的用0填充
        padded_item_sequence = torch.nn.functional.pad(
            torch.tensor(dataset.token2id(dataset.iid_field, item_sequence)),
            (0, pad_length - item_length),
            "constant",
            0,
        )

        '''To perform prediction, we need to create the sequence in this
        interaction object.'''        
        input_interaction = Interaction(
            {
                "aid_list": padded_item_sequence.reshape(1, -1),
                "item_length": torch.tensor([item_length]),
            }
        )
        '''
        In full_sort_predict, first we pass the sequence forward in the model to get the next article.
        This forward pass gives us an embedding. We multiple this embedding with the embedding space 
        learnt by the model. This matrix multiplication gives us a single score for each item. The higher 
        the score, the closer that article is to the predicted embedding. 
        '''
        scores = model.full_sort_predict(input_interaction.to(model.device))
        scores = scores.view(-1, dataset.item_num)
        scores[:, 0] = -np.inf  # pad item score -> -inf

        '''Top 20 scores and items are selected using torch.topk.'''
        topk_score, topk_iid_list = torch.topk(scores, item_history_dict["topk"])

        predicted_score_list = topk_score.tolist()[0]
        '''Predicted items need to be translated back into original article IDs 
        using dataset.id2token.'''
        predicted_item_list = dataset.id2token(
            dataset.iid_field, topk_iid_list.tolist()
        ).tolist()

        recommended_items = {
            "score_list": predicted_score_list,
            "item_list": predicted_item_list,
        }
        return recommended_items


```

上面的调用  fill999之前

```
    print('开始测试gru')
    gru = GRU(fold=fold,cfg=config,name='gru',data_path=config.data_path,max_cands=20)
    gru = gru.load_candidates_file()
    print(f'gru!!! {gru}')
    candidates = candidates.join(gru, on=['session', 'aid'], how='outer')
    del gru
    print('测试完成')
```

gru的配置

```
    gru_model_name = 'GRU4Rec-Jan-24-2023_16-59-54.pth'  # 这个要训出来之后再改成文件夹里的那个名字


    parameter_dict = {
    'data_path': './data/',
    'USER_ID_FIELD':'session',
    'ITEM_ID_FIELD': 'aid',
    'TIME_FIELD': 'ts',
    'user_inter_num_interval': "[5,Inf)",
    'item_inter_num_interval': "[5,Inf)",
    'load_col': {
        'inter': 
            ['session', 'aid', 'ts']
                },
#    'train_neg_sample_args': None,

    'save_dataset':True,
    'save_dataloaders':True,
    # 'dataloaders_save_path':'./data/recbox',
    # 'dataset_save_path':'./data/recbox',
    'checkpoint_dir': './data/recbox/',  

    'epochs': 10,
    'stopping_step':3,
    'loss_type':'BPR',
    'eval_batch_size': 1024,
    #'train_batch_size': 1024,
#    'enable_amp':True,
    'MAX_ITEM_LIST_LENGTH': 20,   #########
    'eval_args': {
        'split': {'RS': [8, 2, 0]},
        'group_by': 'user',
        'order': 'TO',
        'mode': 'full'}
}

```

