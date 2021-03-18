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
model_name = 'indobenchmark/indobert-lite-base-p1'
max_seq_length = 167 # for train and test
preprocessing_num_workers = 4
batch_size= 128

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
        model.resize_token_embeddings(30521)
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
        
#         self.save_hyperparameters()
#         config = AutoConfig.from_pretrained(
#             model_name_or_path=model_name, return_dict=True)
        
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
        # training_step defined the train loop. It is independent of forward
        # barch
        input_ids = batch['input_ids']
        input_ids = self.reshape_(input_ids)
        attention_mask = batch['attention_mask']
        attention_mask = self.reshape_(attention_mask)
        token_type_ids = batch['token_type_ids']
        token_type_ids = self.reshape_(token_type_ids)
        poi_start = batch['POI'][0]
        poi_end = batch['POI'][1]
        street_start = batch['street'][0]
        street_end = batch['street'][1]
        out_cls, embedding = self(input_ids, attention_mask, token_type_ids)
        
        # loss
        cls_loss = F.cross_entropy(out_cls, batch['cls_label'])
        # compute poi loss, where cls label =1 or 3
        poi_mask = batch['cls_label'] %2 == 1
        poi_loss = 0
        for index, poi in enumerate(poi_mask):
            if poi:
                poi_loss += F.cross_entropy(embedding[index,:,0].unsqueeze(dim=0), poi_start[index].unsqueeze(dim=0))
                poi_loss += F.cross_entropy(embedding[index,:,1].unsqueeze(dim=0), poi_end[index].unsqueeze(dim=0))
        # compute street loss, where cls label =2 or 3 (3 is calculated above)
        street_mask = batch['cls_label'] == 2
        street_loss = 0
        for index, street in enumerate(street_mask):
            if street:
                street_loss += F.cross_entropy(embedding[index,:,2].unsqueeze(dim=0), street_start[index].unsqueeze(dim=0))
                street_loss += F.cross_entropy(embedding[index,:,3].unsqueeze(dim=0), street_end[index].unsqueeze(dim=0))

        total_loss = (cls_loss + poi_loss + street_loss)/3 # consider scale cls_loss larger, as found in Squad 2.0 paper
        self.log('train_loss', total_loss, on_step=True, prog_bar=True)
        self.log('cls_loss', cls_loss, on_step=True)
        self.log('poi_loss', poi_loss, on_step=True)
        self.log('street_loss', street_loss, on_step=True)
        return {'loss': total_loss, 'cls': cls_loss, 'poi': poi_loss, 'street': street_loss} # must contrain key loss
#     https://github.com/PyTorchLightning/pytorch-lightning/issues/2783#issuecomment-710615867
    
    def training_epoch_end(self, train_step_outs):
        epoch_train_loss = 0
        epoch_cls_loss = 0
        epoch_poi_loss = 0
        epoch_street_loss = 0
        for d in train_step_outs:
            epoch_train_loss += d['loss']
            epoch_cls_loss += d['cls']
            epoch_poi_loss += d['poi']
            epoch_street_loss += d['street']
        self.log('loss', epoch_train_loss/len(train_step_outs), on_epoch=True, prog_bar=True)
        self.log('poi_start', epoch_cls_loss/len(train_step_outs), on_epoch=True, prog_bar=True)
        self.log('poi_end', epoch_poi_loss/len(train_step_outs), on_epoch=True, prog_bar=True)
        self.log('street_start', epoch_street_loss/len(train_step_outs), on_epoch=True, prog_bar=True)
    
    def validation_step(self, batch, index):
        # batch
        input_ids = batch['input_ids']
        input_ids = self.reshape_(input_ids)
        attention_mask = batch['attention_mask']
        attention_mask = self.reshape_(attention_mask)
        token_type_ids = batch['token_type_ids']
        token_type_ids = self.reshape_(token_type_ids)
        poi_start = batch['POI'][0]
        poi_end = batch['POI'][1]
        street_start = batch['street'][0]
        street_end = batch['street'][1]
        out_cls, embedding = self(input_ids, attention_mask, token_type_ids)
                
        # loss
        cls_loss = F.cross_entropy(out_cls, batch['cls_label'])
        # compute poi loss, where cls label =1 or 3
        poi_mask = batch['cls_label'] %2 == 1
        poi_loss = 0
        for index, poi in enumerate(poi_mask):
            if poi:
                poi_loss += F.cross_entropy(embedding[index,:,0].unsqueeze(dim=0), poi_start[index].unsqueeze(dim=0))
                poi_loss += F.cross_entropy(embedding[index,:,1].unsqueeze(dim=0), poi_end[index].unsqueeze(dim=0))
        # compute street loss, where cls label =2 or 3 (3 is calculated above)
        street_mask = batch['cls_label'] == 2
        street_loss = 0
        for index, street in enumerate(street_mask):
            if street:
                street_loss += F.cross_entropy(embedding[index,:,2].unsqueeze(dim=0), street_start[index].unsqueeze(dim=0))
                street_loss += F.cross_entropy(embedding[index,:,3].unsqueeze(dim=0), street_end[index].unsqueeze(dim=0))

        total_loss = cls_loss + poi_loss + street_loss     

        # acc
        _, cls_ = torch.max(out_cls, dim=1)
        pred_poi_start, pred_poi_end = get_pos(0,1, embedding, cls_)
        pred_street_start, pred_street_end = get_pos(2,3, embedding, cls_)
        
        def get_acc(pred, gt):
            return torch.tensor(accuracy_score(pred.cpu(), gt.cpu()))
        val_accs = [get_acc(pred_poi_start, poi_start), get_acc(pred_poi_end, poi_end), get_acc(pred_street_start, street_start), get_acc(pred_street_end, street_end)]
        
        self.log('val_loss', total_loss, on_step=True)
        self.log('poi_start', val_accs[0], on_step=True)
        self.log('poi_end', val_accs[1], on_step=True)
        self.log('street_start', val_accs[2], on_step=True)
        self.log('street_end', val_accs[3], on_step=True)
        return {'val_loss': total_loss, 'poi_start': val_accs[0], 'poi_end': val_accs[1], 'street_start': val_accs[2], 'street_end': val_accs[3]}
        # may use F1 to measure the performance
    
    def validation_epoch_end(self, valid_step_outs):
        epoch_val_loss = 0
        epoch_accs = [0,0,0,0]
        for d in valid_step_outs:
            epoch_val_loss += d['val_loss']
            epoch_accs[0] += d['poi_start']
            epoch_accs[1] += d['poi_end']
            epoch_accs[2] += d['street_start']
            epoch_accs[3] += d['street_end']
        self.log('val_loss', epoch_val_loss/len(valid_step_outs), on_epoch=True, prog_bar=True)
        self.log('poi_start', epoch_accs[0]/len(valid_step_outs), on_epoch=True, prog_bar=True)
        self.log('poi_end', epoch_accs[1]/len(valid_step_outs), on_epoch=True, prog_bar=True)
        self.log('street_start', epoch_accs[2]/len(valid_step_outs), on_epoch=True, prog_bar=True)
        self.log('street_end', epoch_accs[3]/len(valid_step_outs), on_epoch=True, prog_bar=True)
        
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
        tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p1") # make this global
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
                    current_ret = tokenizer.decode(current_input_ids[start:end], skip_special_tokens=True)
                rets.append(current_ret)
        
        pois = decode_(pred_poi_start, pred_poi_end)
        streets = decode_(pred_street_start, pred_street_end)
        return (pois, streets)
    
    def test_epoch_end(self, test_outs):
        # save file
        answers = []
        for pois, streets in test_outs:
            for poi, street in zip(pois,streets):
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
        self.train_file = 'train.csv'
        self.valid_file = 'train.csv'
        self.test_file = 'test.csv'
        self.batch_size = batch_size
      # When doing distributed training, Datamodules have two optional arguments for
      # granular control over download/prepare/splitting data:            

      # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage=None):
        # step is either 'fit', 'validate', 'test', or 'predict'. 90% of the time not relevant
        # load dataset
        datasets = load_dataset('csv', data_files='train.csv', split=['train[:80%]', 'train[80%:]'])
        column_names = ['id', 'raw_address', 'POI/street']

        tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
        #pad 0 https://huggingface.co/transformers/model_doc/bert.html

        def tokenize_fn(entry):
            encoded = tokenizer.encode(entry['raw_address'])

            # also handle the labels here
            def find_sublist(lst1, lst2):
                lst1 = lst1[1:len(lst1)-1]
                lst2 = lst2[1:len(lst2)-1]
                if len(lst1) == 0 or len(lst2) == 0:
                    return (0, 0)
                for i in range(len(lst2)-len(lst1)+1):
                    if lst2[i:i+len(lst1)] == lst1:
#                         return i+1, i+len(lst1)+1
                        return i, i+len(lst1) # [TODO] debug on this plus 1 due to splitting [CLS] at start of sequence
                else:
                    return (0, 0) # -1 triggers index out of bound error
            labels = entry['POI/street'].split('/')
            encoded_poi = tokenizer.encode(labels[0])
            entry_poi_pos = find_sublist(encoded_poi, encoded)
            encoded_street = tokenizer.encode(labels[1])
            entry_street_pos = find_sublist(encoded_street, encoded)
            cls_label = 0
            if labels[0]:
                cls_label += 1
            if labels[1]:
                cls_label += 2
            return {'POI':entry_poi_pos, 'street': entry_street_pos, 'cls_label': cls_label}
        datasets = [dataset.map(lambda entries: tokenizer(entries['raw_address'], padding=True), batched=True, batch_size=batch_size, num_proc=1) for dataset in datasets]
        tokenized_d_train, tokenized_d_valid = [dataset.map(tokenize_fn, num_proc=preprocessing_num_workers) for dataset in datasets] # attempts to avoid size mismatch 
        self.train_dataset = tokenized_d_train
        self.valid_dataset = tokenized_d_valid
        
        # test dataset
        test_d = load_dataset('csv', data_files='test.csv')
        tokenized_d_test = test_d.map(lambda entries: tokenizer(entries['raw_address'], padding=True), batched=True, batch_size=batch_size, num_proc=1)
        self.test_dataset = tokenized_d_test['train'] # named by the dataset module
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=1, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=1, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=1)

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='.',
    filename='bert-ind-{epoch:02d}-{val_loss:.2f}',
    save_top_k=10,
    mode='min',
)

dm = Dm()

lm = My_lm()
# trainer = pl.Trainer(gpus=1, overfit_batches=1)
# trainer = pl.Trainer(gpus=1, fast_dev_run=True)# , profiler='simple')
trainer = pl.Trainer(gpus=1, max_epochs=1, callbacks=[checkpoint_callback])
# trainer = pl.Trainer(gpus=1, max_epochs=10, limit_train_batches=10, limit_val_batches=3, callbacks=[checkpoint_callback])
# trainer = pl.Trainer(gpus=1, max_epochs=100)
trainer.fit(lm,dm)
result = trainer.test()

# test with 1 epoch
# then test with 100 epochs