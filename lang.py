import os
import torch
from datasets import load_dataset
from torch import nn
import torch.nn.functional as F
from transformers import (AutoModel, BertTokenizer, AutoConfig, DataCollatorForLanguageModeling) # AutoModelForMaskedLM
from torch.utils.data import DataLoader, random_split #  AutoTokenizer
from torchvision import transforms
import pytorch_lightning as pl
import gzip
import csv
from sklearn.metrics import accuracy_score
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Configs
model_name = 'indobenchmark/indobert-lite-base-p1'
max_seq_length = 167
preprocessing_num_workers = 4
batch_size= 16

class IndBert(nn.Module):
    def __init__(self):
        super(IndBert, self).__init__()
        model = AutoModel.from_pretrained(model_name)
        model.resize_token_embeddings(30521) # len(tokenizer)
        self.bert = model
        self.linear = nn.Linear(in_features=768, out_features=4, bias=True)
        # 4 for poi start and end, street start and end
    def forward(self, input_ids, attention_mask, token_type_ids):
#         print(input_ids)
#         print('-------------')
#         print(attention_mask)
#         print('......................')
#         print(token_type_ids)
        out = self.bert(input_ids, attention_mask, token_type_ids)
        out = self.linear(out[0])
        return out

class My_lm(pl.LightningModule):

    def __init__(self):
        super().__init__()
        
#         self.save_hyperparameters()
        
#         config = AutoConfig.from_pretrained(
#             model_name_or_path=model_name, return_dict=True)
        
        self.model = IndBert()
        

    def forward(self, input_ids, attention_mask, token_type_ids):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.model(input_ids, attention_mask, token_type_ids)
#         print('embedding')
#         print(embedding)
        return embedding
    
    def reshape_(self, x):
        _x = torch.stack(x,dim=0)
        _x = torch.transpose(_x , 0, 1)
        return _x
    
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
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
        preds = self(input_ids, attention_mask, token_type_ids)
        loss = F.cross_entropy(preds[:,:,0], poi_start) + F.cross_entropy(preds[:,:,1], poi_end) + F.cross_entropy(preds[:,:,2], street_start) + F.cross_entropy(preds[:,:,3], street_end)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return {'loss': loss} # must contrain key loss
#     https://github.com/PyTorchLightning/pytorch-lightning/issues/2783#issuecomment-710615867
    
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
        preds = self(input_ids, attention_mask, token_type_ids)
        
#         # loss
        loss = F.cross_entropy(preds[:,:,0], poi_start) + F.cross_entropy(preds[:,:,1], poi_end) + F.cross_entropy(preds[:,:,2], street_start) + F.cross_entropy(preds[:,:,3], street_end)
        
#         # acc
        val0, pos0 = torch.max(preds[:,:,0], dim=1)
        val1, pos1 = torch.max(preds[:,:,1], dim=1)
        val2, pos2 = torch.max(preds[:,:,2], dim=1)
        val3, pos3 = torch.max(preds[:,:,3], dim=1)
        val_acc0 = accuracy_score(pos0.cpu(), poi_start.cpu())
        val_acc0 = torch.tensor(val_acc0)
        val_acc1 = accuracy_score(pos1.cpu(), poi_end.cpu())
        val_acc1 = torch.tensor(val_acc1)
        val_acc2 = accuracy_score(pos2.cpu(), street_start.cpu())
        val_acc2 = torch.tensor(val_acc2)
        val_acc3 = accuracy_score(pos3.cpu(), street_end.cpu())
        val_acc3 = torch.tensor(val_acc3)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('poi_start', val_acc0, on_step=True, on_epoch=True, prog_bar=True)
        self.log('poi_end', val_acc1, on_step=True, on_epoch=True, prog_bar=True)
        self.log('street_start', val_acc2, on_step=True, on_epoch=True, prog_bar=True)
        self.log('street_end', val_acc3, on_step=True, on_epoch=True, prog_bar=True)
        return {'loss': loss, 'val_acc0': val_acc0, 'val_acc1': val_acc1, 'val_acc2': val_acc2, 'val_acc3': val_acc3}
    
    def test_step(self, batch, batch_idx):
        pass
    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=2e-04, eps=1e-08)
#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
#         return optimizer
# Which loss fn ?

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
                        return i+1, i+len(lst1)+1
                else:
                    return (0, 0) # -1 triggers index out of bound error
            labels = entry['POI/street'].split('/')
            encoded_poi = tokenizer.encode(labels[0])
        #     print('labels', labels[0])
        #     print(encoded_poi)
            entry_poi_pos = find_sublist(encoded_poi, encoded)
            encoded_street = tokenizer.encode(labels[1])
            entry_street_pos = find_sublist(encoded_street, encoded)
            return {'POI':entry_poi_pos, 'street': entry_street_pos}
        datasets = [dataset.map(lambda entries: tokenizer(entries['raw_address'], padding=True), batched=True) for dataset in datasets]
        tokenized_d_train, tokenized_d_valid = [dataset.map(tokenize_fn) for dataset in datasets]
        self.train_dataset = tokenized_d_train
        self.valid_dataset = tokenized_d_valid
      # return the dataloader for each split
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        pass

dm = Dm()

lm = My_lm()
# trainer = pl.Trainer(gpus=1, max_epochs=1, limit_train_batches=10, limit_val_batches=3)
trainer = pl.Trainer(gpus=1, max_epochs=100)
trainer.fit(lm,dm)