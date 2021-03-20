#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import torch
from datasets import load_dataset
from torch import nn
import torch.nn.functional as F
from transformers import (AutoModel, BertTokenizer, AutoConfig, EncoderDecoderModel, DataCollatorForLanguageModeling) # AutoModelForMaskedLM
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


# In[3]:


# Configs
model_name = "indolem/indobert-base-uncased" #'indobenchmark/indobert-lite-base-p1'
max_seq_length = 167 # for train and test
preprocessing_num_workers = 4
batch_size=256 # depend on gpu memory
test_batch_size=2048 # speed up

tokenizer = BertTokenizer.from_pretrained(model_name) # make this global
test_file = '20210320-231710.csv'


# In[4]:


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
model = SCBert()
s_d = torch.load('bert-ind-epoch=006-val_loss=0.002.ckpt')
new_d = {}
for k,v in s_d['state_dict'].items():
    new_k = '.'.join(k.split('.')[2:])
    new_d[new_k] = v
model.bert.load_state_dict(new_d)
# model.bert


# In[5]:


df = pd.read_csv('20210320-231710.csv')
ps = df['POI/street']
p_s = []
s_s = []
for i in ps:
    p, s = i.split('/')
    p_s.append(p)
    s_s.append(s)


# In[8]:


# model.cuda()


# In[9]:


ans_p = []
count = 0
for i in p_s:
    count += 1
    if count % 1000 == 1:
        print(count)
    if len(i) == 0:
        ans_p.append(i)
    else:
        # predict
        enc = torch.tensor(tokenizer.encode(i)).unsqueeze(0).cuda()
        out = model.bert.generate(enc, decoder_start_token_id=model.bert.config.decoder.pad_token_id) #self.generate(input_ids, decoder_start_token_id=model.config.decoder.pad_token_id)
        ret = tokenizer.decode(out[0].cpu(), skip_special_tokens=True)
        ans_p.append(ret)


# In[ ]:


ans_s = []
for i in s_s:
    if len(i) == 0:
        ans_s.append(i)
    else:
        # predict
        enc = torch.tensor(tokenizer.encode(i)).unsqueeze(0).cuda()
        out = model.bert.generate(enc, decoder_start_token_id=model.bert.config.decoder.pad_token_id) #self.generate(input_ids, decoder_start_token_id=model.config.decoder.pad_token_id) 
        ret = tokenizer.decode(out[0].cpu(), skip_special_tokens=True)
        ans_s.append(ret)


# In[ ]:


df_answer = pd.DataFrame({'id': list(range(len(ans_s))), 'POI/street': [p+'/'+s for p, s in zip(ans_p, ans_s)]})
filename1 = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
df_answer.to_csv(filename1+'.csv', index=False)


# In[ ]:




