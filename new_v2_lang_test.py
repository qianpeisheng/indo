import os
import torch
from datasets import load_dataset
from torch import nn
import torch.nn.functional as F
from transformers import (AutoModel, BertTokenizer, AutoConfig, DataCollatorForLanguageModeling) # AutoModelForMaskedLM
from torch.utils.data import DataLoader, random_split #  AutoTokenizer
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from collections import Counter

import gzip
import csv
from sklearn.metrics import accuracy_score
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

import pandas as pd
import datetime

# Configs
model_name = "indolem/indobert-base-uncased" #'indobenchmark/indobert-lite-base-p1'
max_seq_length = 167 # for train and test
preprocessing_num_workers = 4
batch_size=256 # depend on gpu memory
test_batch_size=2048 # speed up

tokenizer = BertTokenizer.from_pretrained(model_name) # make this global

# utils
def get_pos(index1, index2, embedding, cls_):
    val1, pos1 = torch.max(embedding[:,:,index1], dim=1)
    val2, pos2 = torch.max(embedding[:,:,index2], dim=1)
    for i, v in enumerate(cls_):
        if index2 < 2: # poi
            if v == 0 or v == 2:
                pos1[i] = 0
                pos2[i] = 0
        else: # street
            if v == 0 or v == 1:
                pos1[i] = 0
                pos2[i] = 0
    return pos1, pos2

# out[0] **is** out.last_hidden_state
class IndBert(nn.Module):
    def __init__(self):
        super(IndBert, self).__init__()
        model = AutoModel.from_pretrained(model_name)
        # model.resize_token_embeddings(30521)
        model.resize_token_embeddings(len(tokenizer))
# https://github.com/huggingface/transformers/issues/4153
        self.bert = model
        self.linear = nn.Linear(in_features=768, out_features=4, bias=True)
        # 4 for poi start and end, street start and end
        self.linear_cls = nn.Linear(in_features=768, out_features=4, bias=True)
    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(input_ids, attention_mask, token_type_ids)# (hidden_state, pooled_out)
        out_sentence = out.last_hidden_state[:,1:,:]
        out_cls = out.last_hidden_state[:,0,:]
        out_cls = self.linear_cls(out_cls)
        out_sentence = self.linear(out_sentence)
        return out_cls, out_sentence

class My_lm(pl.LightningModule):

    def __init__(self):
        super().__init__()
        
        self.model = IndBert()
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        # in lightning, forward defines the prediction/inference actions
        out_cls, embedding = self.model(input_ids, attention_mask, token_type_ids)
        return out_cls, embedding
    
    def reshape_(self, x):
        _x = torch.stack(x,dim=0)
        _x = torch.transpose(_x , 0, 1)
        return _x
    
    def training_step(self, batch, batch_idx):
        pass
    
    def training_epoch_end(self, train_step_outs):
        pass
    
    def validation_step(self, batch, index):
        pass
                
    
    def validation_epoch_end(self, valid_step_outs):
        pass

    def test_step(self, batch, batch_idx):
        # batch
        input_ids = batch['input_ids']
        input_ids = self.reshape_(input_ids)
        attention_mask = batch['attention_mask']
        attention_mask = self.reshape_(attention_mask)
        token_type_ids = batch['token_type_ids']
        token_type_ids = self.reshape_(token_type_ids)
        out_cls, embedding = self(input_ids, attention_mask, token_type_ids)
        # map ids to tokens
        _, cls_ = torch.max(out_cls, dim=1)
        pred_poi_start, pred_poi_end = get_pos(0,1, embedding, cls_)
        pred_street_start, pred_street_end = get_pos(2,3, embedding, cls_)
        tokenizer = BertTokenizer.from_pretrained(model_name) # make this global
        def decode_(pred_start, pred_end):   
            rets = []
            for index, (start, end) in enumerate(zip(pred_start, pred_end)):
                if start == 0 and end == 0:
                    current_ret = ''
                else:
                # limit end to the length of the input, to avoid [SEP] and [PAD]
                # Note that decoder skips special tokens, so end may > len(input_ids)
                    current_input_ids = input_ids[index]
                    end = min(len(current_input_ids), end)
                    current_ret = tokenizer.decode(current_input_ids[start+1:end+1], skip_special_tokens=True)
                    # NOTE [IMPORTANT] This +1 is the key to raise score from 0.01 to 0.5x
                rets.append(current_ret)
            return rets
        pois = decode_(pred_poi_start, pred_poi_end)
        streets = decode_(pred_street_start, pred_street_end)
        return (pois, streets)
    
    def test_epoch_end(self, test_outs):
        # save file
        answers = []
        for pois, streets in test_outs:
            for poi, street in zip(pois, streets):
                answers.append(poi+'/'+street)
        df_answer = pd.DataFrame({'id': range(len(answers)), 'POI/street': answers})
        filename1 = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        df_answer.to_csv(filename1+'.csv', index=False)
        return filename1
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-05, eps=1e-08) # check requires grad
        # 3e-5

class Dm(pl.LightningDataModule):
    def __init__(self, batch_size=batch_size):
        super().__init__()
        self.test_file = 'new_test_2.csv'
        self.batch_size = batch_size
      # When doing distributed training, Datamodules have two optional arguments for
      # granular control over download/prepare/splitting data:            

      # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage=None):
        # step is either 'fit', 'validate', 'test', or 'predict'. 90% of the time not relevant
        # load dataset

        tokenizer = BertTokenizer.from_pretrained(model_name)
        #pad 0 https://huggingface.co/transformers/model_doc/bert.html
        
        # test dataset
        test_d = load_dataset('csv', data_files=self.test_file, split='train[:100%]') # adjust the ratio for debugging
        tokenized_d_test = test_d.map(lambda entries: tokenizer(entries['raw_address'], padding=True), batched=True, batch_size=test_batch_size, num_proc=1)
        self.test_dataset = tokenized_d_test# ['train'] # named by the dataset module
        
    def train_dataloader(self):
        pass
    def val_dataloader(self):
        pass

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=test_batch_size, num_workers=1)

# checkpoint_callback = ModelCheckpoint(
#     # monitor='val_loss',
#     monitor='avg_acc',
#     dirpath='./exp2/',
#     filename='bert-ind-{epoch:03d}-{val_loss:.3f}-{avg_acc:.3f}',
#     save_top_k=100,
#     mode='max',
# )

dm = Dm()
lm = My_lm()

# # debug
# trainer = pl.Trainer(gpus=1, overfit_batches=1)
# trainer = pl.Trainer(gpus=1, fast_dev_run=True)# , profiler='simple')
# trainer = pl.Trainer(gpus=1, max_epochs=1, callbacks=[checkpoint_callback])
# trainer = pl.Trainer(gpus=1, max_epochs=10, limit_train_batches=10, limit_val_batches=3, callbacks=[checkpoint_callback])

# # standard train, validation and test
# trainer = pl.Trainer(gpus=1, max_epochs=100, callbacks=[checkpoint_callback], stochastic_weight_avg=True, gradient_clip_val=1)
# trainer.fit(lm,dm)
# result = trainer.test()

# # testing only 
# use larger batch size to speed up testing
dm.setup()
model = lm.load_from_checkpoint('exp1/bert-ind-epoch=029-val_loss=0.285-avg_acc=0.925.ckpt')
trainer = pl.Trainer(gpus=1)
result = trainer.test(model, test_dataloaders=dm.test_dataloader())
