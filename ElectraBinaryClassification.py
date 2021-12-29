import os

import torch
import torchmetrics
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import transformers
from transformers import ElectraForSequenceClassification, ElectraTokenizer, AdamW

device = torch.device("cuda")

# https://medium.com/huggingface/multi-label-text-classification-using-bert-the-mighty-transformer-69714fa3fb3d
# https://huggingface.co/docs/transformers/v4.15.0/en/model_doc/electra#transformers.ElectraForSequenceClassification

class ElectraClassification(pl.LightningModule) :
    def __init__(self, learning_rate) :
        super().__init__()
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        
        self.electra = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-small-v3-discriminator")

        self.metric_acc = torchmetrics.Accuracy()
        self.metric_f1 = torchmetrics.F1(num_classes=2)
        self.metric_rec = torchmetrics.Recall(num_classes=2)
        self.metric_pre = torchmetrics.Precision(num_classses=2)

        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None) :
        output = self.electra(input_ids=input_ids, 
                                attention_mask=attention_mask, 
                                labels=labels)
        return output

    def training_step(self, batch, batch_idx) :
        '''
        ##########################################################
        electra forward input shape information
        * input_ids.shape (batch_size, max_length)
        * attention_mask.shape (batch_size, max_length)
        * label.shape (batch_size,)
        ##########################################################
        '''

        # change label shape (list -> torch.Tensor((batch_size, 1)))
        label = batch['label'].view([-1,1])

        output = self(input_ids=batch['input_ids'].to(device),
                        attention_mask=batch['attention_mask'].to(device),
                        labels=label.to(device))
        '''
        ##########################################################
        electra forward output shape information
        * loss.shape (1,)
        * logits.shape (batch_size, config.num_labels=2)
        '''
        logits = output.logits

        loss = output.loss
        # loss = self.loss_func(logits.to(device), batch['label'].to(device))

        softmax = nn.functional.softmax(logits, dim=1)
        preds = softmax.argmax(dim=1)

        self.log("train_loss", loss, prog_bar=True)
        
        return {
            'loss' : loss,
            'pred' : preds,
            'label' : batch['label']
        }

    def training_epoch_end(self, outputs, state='train') :
        y_true = []
        y_pred = []
        for i in outputs :
            y_true += i['label'].tolist()
            y_pred += i['pred'].tolist()

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        # self.log(state+'_acc', acc, on_epoch=True, prog_bar=True)
        # self.log(state+'_precision', prec, on_epoch=True, prog_bar=True)
        # self.log(state+'_recall', rec, on_epoch=True, prog_bar=True)
        # self.log(state+'_f1', f1, on_epoch=True, prog_bar=True)
        print(f'[Epoch {self.trainer.current_epoch} {state.upper()}] Acc: {acc}, Prec: {prec}, Rec: {rec}, F1: {f1}')

    def validation_step(self, batch, batch_idx) :
        '''
        ##########################################################
        electra forward input shape information
        * input_ids.shape (batch_size, max_length)
        * attention_mask.shape (batch_size, max_length)
        ##########################################################
        '''
        output = self(input_ids=batch['input_ids'].to(device),
                        attention_mask=batch['attention_mask'].to(device))
        logits = output.logits
        preds = nn.functional.softmax(logits, dim=1).argmax(dim=1)

        labels = batch['label']
        accuracy = self.metric_acc(preds, labels)
        f1 = self.metric_f1(preds, labels)
        recall = self.metric_rec(preds, labels)
        precision = self.metric_pre(preds, labels)
        self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True)
        self.log('val_f1', f1, on_epoch=True, prog_bar=True)
        self.log('val_recall', recall, on_epoch=True, prog_bar=True)
        self.log('val_precision', precision, on_epoch=True, prog_bar=True)

        return {
            'accuracy' : accuracy,
            'f1' : f1,
            'recall' : recall,
            'precision' : precision
        }

    def validation_epoch_end(self, outputs) :
        val_acc = torch.stack([i['accuracy'] for i in outputs]).mean()
        val_f1 = torch.stack([i['f1'] for i in outputs]).mean()
        val_rec = torch.stack([i['recall'] for i in outputs]).mean()
        val_pre = torch.stack([i['precision'] for i in outputs]).mean()
        # self.log('val_f1', val_f1, on_epoch=True, prog_bar=True)
        # self.log('val_acc', val_acc, on_epoch=True, prog_bar=True)
        print(f'val_accuracy : {val_acc}, val_f1 : {val_f1}, val_recall : {val_rec}, val_precision : {val_pre}')
        
    
    def configure_optimizers(self) :
        optimizer = torch.optim.AdamW(self.electra.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        
        return {
            'optimizer' : optimizer,
            'lr_scheduler' : lr_scheduler
        }