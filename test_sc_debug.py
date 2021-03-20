#!/usr/bin/env python
# coding: utf-8
import os
import torch
from datasets import load_dataset
from torch import nn
import torch.nn.functional as F
from transformers import (AutoModel, BertTokenizer, AutoConfig, DataCollatorForLanguageModeling, EncoderDecoderModel) # AutoModelForMaskedLM
from torch.utils.data import DataLoader, random_split #  AutoTokenizer
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

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
model_name = 'indolem/indobert-base-uncased'#'indobenchmark/indobert-lite-base-p1'
max_seq_length = 167 # for train and test
preprocessing_num_workers = 4
batch_size= 128
tokenizer = BertTokenizer.from_pretrained(model_name)

# out[0] **is** out.last_hidden_state
class SCBert(nn.Module):
    def __init__(self):
        super(SCBert, self).__init__()
        model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_name, model_name) #("indolem/indobert-base-uncased", "indolem/indobert-base-uncased")
#         model.resize_token_embeddings(len(tokenizer))
# https://github.com/huggingface/transformers/issues/4153
        self.bert = model
    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, labels=labels)# (hidden_state, pooled_out)
        return out
# outputs = model(input_ids=input_ids, decoder_input_ids=input_ids, labels=decoder_ids)
class My_lm(pl.LightningModule):

    def __init__(self):
        super().__init__()
        
#         self.save_hyperparameters()
#         config = AutoConfig.from_pretrained(
#             model_name_or_path=model_name, return_dict=True)
        
        self.model = SCBert()
        
    def forward(self,input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels):
        # in lightning, forward defines the prediction/inference actions
        out = self.model(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels)
        return out
    
    def reshape_(self, x):
        _x = torch.stack(x,dim=0)
        _x = torch.transpose(_x , 0, 1)
        return _x
    
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        # barch
        input_ids = batch['input_ids']
        input_ids = self.reshape_(input_ids)
        attention_mask = batch['attention_mask']
        attention_mask = self.reshape_(attention_mask)
        decoder_ids = batch['decoder_ids']
        decoder_ids = self.reshape_(decoder_ids)
        outputs = self(input_ids, attention_mask, input_ids, attention_mask, decoder_ids)
        
        # loss
        loss, logits = outputs.loss, outputs.logits

        self.log('train_loss', loss, on_step=True, prog_bar=True)
        return {'loss': loss} # must contrain key loss
#     https://github.com/PyTorchLightning/pytorch-lightning/issues/2783#issuecomment-710615867
    
    def training_epoch_end(self, train_step_outs):
        epoch_train_loss = 0
        for d in train_step_outs:
            epoch_train_loss += d['loss']
        self.log('loss', epoch_train_loss/len(train_step_outs), on_epoch=True, prog_bar=True)
    
    def validation_step(self, batch, index):
        # batch
        input_ids = batch['input_ids']
        input_ids = self.reshape_(input_ids)
        attention_mask = batch['attention_mask']
        attention_mask = self.reshape_(attention_mask)
        decoder_ids = batch['decoder_ids']
        decoder_ids = self.reshape_(decoder_ids)
        outputs = self(input_ids, attention_mask, input_ids, attention_mask, decoder_ids)
                
        # loss
        loss, logits = outputs.loss, outputs.logits

        # acc
        
        self.log('val_loss', loss, on_step=True)
        return {'val_loss': loss}
        # may use F1 to measure the performance
    
    def validation_epoch_end(self, valid_step_outs):
        epoch_val_loss = 0
        for d in valid_step_outs:
            epoch_val_loss += d['val_loss']
        self.log('val_loss', epoch_val_loss/len(valid_step_outs), on_epoch=True, prog_bar=True)
        
    def test_step(self, batch, batch_idx):
        # batch
        input_ids = batch['input_ids']
        input_ids = self.reshape_(input_ids)
        attention_mask = batch['attention_mask']
        attention_mask = self.reshape_(attention_mask)
        decoder_ids = batch['decoder_ids']
        decoder_ids = self.reshape_(decoder_ids)
        outputs = self(input_ids, attention_mask, input_ids, attention_mask, decoder_ids)
        generated = self.generate(input_ids, decoder_start_token_id=model.config.decoder.pad_token_id)
        # map ids to tokens
        rets = []
        for i in generated:
            current_ret = tokenizer.decode(i, skip_special_tokens=True)
            rets.append(current_ret)      
        return rets
    
    def test_epoch_end(self, test_outs):
        # save file
        answers = []
        for rets in test_outs:
            for ret in rets:
                answers.append(ret)
        df_answer = pd.DataFrame({'id': range(len(answers)), 'POI/street': answers})
        filename1 = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        df_answer.to_csv(filename1+'.csv', index=False)
        return filename1
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-05, eps=1e-08) # check requires grad
        # Not pretrained in some layers!

class Dm(pl.LightningDataModule):
    def __init__(self, batch_size=batch_size):
        super().__init__()
        self.train_file = 'new_sc.csv'
        self.valid_file = 'new_sc.csv'
        self.test_file = 'new_sc.csv'
        self.batch_size = batch_size
      # When doing distributed training, Datamodules have two optional arguments for
      # granular control over download/prepare/splitting data:            

      # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage=None):
        # step is either 'fit', 'validate', 'test', or 'predict'. 90% of the time not relevant
        datasets = load_dataset('csv', data_files=self.train_file, split=['train[:100%]', 'train[80%:]'])

        #pad 0 https://huggingface.co/transformers/model_doc/bert.html
        def add_decoder_id(entry):
            return {'decoder_ids': entry['input_ids'], 'decoder_attention_mask': entry['attention_mask']}

        # datasets = [dataset.map(lambda entries: tokenizer(['nan' if not d else d for d in entries['label']], padding=True), batched=True, batch_size=batch_size,) for dataset in datasets]
        # datasets = [dataset.map(add_decoder_id, num_proc=preprocessing_num_workers) for dataset in datasets]
        # datasets = [dataset.map(lambda entries: tokenizer(['nan' if not d else d for d in entries['label']], padding=True), batched=True, batch_size=batch_size,) for dataset in datasets]
        # tokenized_d_train, tokenized_d_valid = datasets
        test_d = load_dataset('csv', data_files=self.test_file, split='train[:100%]') # adjust the ratio for debugging
        tokenized_d_test = test_d.map(lambda entries: tokenizer(entries['raw_address'], padding=True), batched=True, batch_size=test_batch_size, num_proc=1)
        self.train_dataset = tokenized_d_train
        self.valid_dataset = tokenized_d_valid
#         self.test_dataset = tokenized_d_test
        
    def train_dataloader(self):
        # return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=1, drop_last=True)
        pass

    def val_dataloader(self):
        # return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=1, drop_last=True)
        pass

    def test_dataloader(self):
#         return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=1)
        pass

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='./new_sc/',
    filename='bert-ind-{epoch:03d}-{val_loss:.3f}',
    save_top_k=100,
    mode='min',
)

dm = Dm()

lm = My_lm()
# trainer = pl.Trainer(gpus=1, overfit_batches=1)
# trainer = pl.Trainer(gpus=1, fast_dev_run=True)# , profiler='simple')
trainer = pl.Trainer(gpus=1, max_epochs=100, callbacks=[checkpoint_callback], stochastic_weight_avg=True, gradient_clip_val=1)
# trainer = pl.Trainer(gpus=1, max_epochs=2, limit_train_batches=100, limit_val_batches=30, callbacks=[checkpoint_callback])
# trainer = pl.Trainer(gpus=1, max_epochs=100)
trainer.fit(lm,dm)
# result = trainer.test()




