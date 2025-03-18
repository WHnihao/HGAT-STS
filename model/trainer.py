import sklearn
import os
from argparse import Namespace
import pickle
import torch
import gc
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data.sampler import SubsetRandomSampler
from pytorch_lightning.core.lightning import LightningModule
from utils.data_loader import collate_fn, Dataset
from pytorch_lightning import Trainer
from utils.torch_utils import f1_score, acc_score, f1c_score
from model.model import Graph_STS
from utils.data_reader import load_dataset, load_dataset_c, get_original_data



class Train_GraphSTS(LightningModule):
    """
    Pytorch-lightning wrapper class for training
    """
   
    def __init__(self, index_epoch, config=None):
        super().__init__()
        self.config = config
        trn_data, val_data, tst_data, self.vocab, phrase_list_repet, phrase_list_pos_repet, vocab_ph, vocab_sent, v_list, s_list, p_list, embed_dic_sent, embed_dic_phrase, embed_dic_word, phrase_dic_dic = load_dataset()

        self.ds_trn, self.ds_val, self.ds_tst = Dataset(trn_data, self.vocab, phrase_list_repet, phrase_list_pos_repet, vocab_ph, vocab_sent, v_list, s_list, p_list), Dataset(val_data, self.vocab, phrase_list_repet, phrase_list_pos_repet, vocab_ph, vocab_sent, v_list, s_list, p_list), Dataset(tst_data, self.vocab, phrase_list_repet, phrase_list_pos_repet, vocab_ph, vocab_sent, v_list, s_list, p_list)
        #print(ds_trn)
        val_data_c, test_data_c = load_dataset_c()
        
        self.ds_val_c, self.ds_tst_c = Dataset(val_data_c, self.vocab, phrase_list_repet, phrase_list_pos_repet, vocab_ph, vocab_sent, v_list, s_list, p_list), Dataset(test_data_c, self.vocab, phrase_list_repet, phrase_list_pos_repet, vocab_ph, vocab_sent, v_list, s_list, p_list)
        exclude = ['np','torch','random','args', 'os', 'argparse', 'parser', 'Namespace', 'sys']
        self.hparams = Namespace(**{k:v for k,v in config.__dict__.items() if k[:2]!='__' and k not in exclude})
        #self.index_epoch = index_epoch
        self.model = Graph_STS(config, self.vocab, index_epoch, vocab_ph, vocab_sent, embed_dic_sent, embed_dic_phrase, embed_dic_word, phrase_dic_dic)
        self.loss_fn = nn.CrossEntropyLoss()#nn.BCELoss()#nn.BCEWithLogitsLoss() # nn.BCELoss()
        #self.softmax = torch.softmax(dim=0)
        self.f1_metric = f1_score
        self.f1c_metric = f1c_score
        self.acc_metric = acc_score
    def simcse_unsup_loss(self, y_pred, device, temp=0.05):
        """无监督的损失函数
        y_pred (tensor): bert的输出, [batch_size * 2, 768]

        """
        # 得到y_pred对应的label, [1, 0, 3, 2, ..., batch_size-1, batch_size-2]
        
        y_true = torch.arange(y_pred.shape[0], device=device)
        
        y_true = (y_true - y_true % 2 * 2) + 1
        # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
        sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
        # 将相似度矩阵对角线置为很小的值, 消除自身的影响
        sim = sim - torch.eye(y_pred.shape[0], device=device) * 1e12
        # 相似度矩阵除以温度系数
        sim = sim / temp
        # 计算相似度矩阵与y_true的交叉熵损失
        # 计算交叉熵，每个case都会计算与其他case的相似度得分，得到一个得分向量，目的是使得该得分向量中正样本的得分最高，负样本的得分最低
        ####print(y_true.shape)
        loss = F.cross_entropy(sim, y_true)
        ####print(loss)
        return torch.mean(loss)
    def simcse_sup_loss(self, y_pred, device, lamda=0.05):
        """
        有监督损失函数
        """
        similarities = F.cosine_similarity(y_pred.unsqueeze(0), y_pred.unsqueeze(1), dim=2)
        row = torch.arange(0, y_pred.shape[0], 3)#3改成2
        col = torch.arange(0, y_pred.shape[0])

        col = col[col % 3 != 0]#3改成2
        similarities = similarities[row, :]
        similarities = similarities[:, col]
        similarities = similarities / lamda
        y_true = torch.arange(0, len(col), 2, device=device)
        loss = F.cross_entropy(similarities, y_true)
        return loss
    
    def configure_optimizers(self):
        if self.config.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        elif self.config.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.config.lr)
        elif self.config.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr)
        elif self.config.optimizer == 'adamax':
            optimizer = torch.optim.Adamax(self.parameters(), lr=self.config.lr)
        if len(self.config.scheduler) == 0:
            return optimizer
        elif self.config.scheduler == 'exp':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.config.lr_decay)
            return [optimizer], [scheduler]
        elif self.config.scheduler == 'cyclic':
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, self.config.base_lr, self.config.max_lr, cycle_momentum=False)
            return [optimizer], [scheduler]

    def forward(self, batch):
        #index_epoch = index_epoch+1
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        pred, phrase_emb, reg_loss = self(batch)   #, pred_ph 
        if batch_idx % 1000 == 0:
            #torch.cuda.empty_cache()
            gc.collect()
            print("pred", batch_idx)

        a = phrase_emb.shape[0]
        #print(a)
        loss1 = self.simcse_unsup_loss(phrase_emb, self.config.device)
        #print(loss1)
        loss1 = loss1/a*64

        rids_p = batch['rids_p']
        phrase_ind = []
        for ind, indexp in enumerate(batch['phrase_ind']):
            if indexp == 1:
                phrase_ind.append(ind)
        phrase_ind_l = torch.LongTensor(phrase_ind)
        #print(rids_p.shape, phrase_ind_l)
        rids_p = rids_p[phrase_ind_l,:]

        rids = torch.argmax(batch['rids'], dim=1)

        rids_p_ph = torch.argmax(rids_p, dim=1)
        #for i , label in enumerate(pred_list):
        rids = torch.argmax(batch['rids'], dim=1)

        loss = self.loss_fn(pred, rids)

        tb_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tb_logs}

    def validation_step(self, *args):
        batch, batch_idx = args
        pred, phrase_emb, reg_loss = self(batch)#, pred_ph
 
        rids = torch.argmax(batch['rids'], dim=1)
        #print(rids)
        
        loss = self.loss_fn(pred, rids)
        loss1 = self.simcse_unsup_loss(phrase_emb, self.config.device)
        ####print(pred)
        #loss = loss2
        pred = self.softmax(pred)#torch.sigmoid(pred)
        #print(batch['utter_index'], batch['conv_batch'])
        pred_list = pred.tolist()
        rids_list = batch['rids'].tolist()
        #for i , label in enumerate(pred_list):
            
           # print(pred_list[i], rids_list[i])

        #print(rids_list)
        return {'val_loss': loss, 
                'pred': pred.detach().cpu().numpy(), 
                'label': batch['rids'].detach().cpu().numpy(),
               }


    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        pred = [oo  for o in outputs for oo in o['pred']]
        label = [oo for o in outputs for oo in o['label']]

        acc = self.acc_metric(label, pred)
        acc = torch.tensor(acc, device=self.config.device)
        tensorboard_logs = {
            'val_loss': avg_loss, 
            'acc': acc
        }

        return {'progress_bar': tensorboard_logs, 
                'log': tensorboard_logs, 
                'val_loss': avg_loss, 
                'acc': acc, 
                #'eval_T2': eval_T2, 
                #'precision': precision, 
                #'recall': recall,
               }


    def test_step(self, *args):
        batch, batch_idx, dl_idx = args
        pred, phrase_emb, reg_loss = self(batch)   # pred_ph,
        rids = torch.argmax(batch['rids'], dim=1)
        loss1 = self.simcse_unsup_loss(phrase_emb, self.config.device)
        loss = self.loss_fn(pred, rids)
        #loss = loss2
        pred = self.softmax(pred)#torch.sigmoid(pred)
        return {'test_loss': loss, 
                'pred': pred.detach().cpu().numpy(), 
                'label': batch['rids'].detach().cpu().numpy(),
               }


    def test_epoch_end(self, outputs):
        for i, output in enumerate(outputs):
            if i == 0:
                dev_loss = torch.stack([x['test_loss'] for x in output]).mean()
                pred = [oo  for o in output for oo in o['pred']]
                label = [oo for o in output for oo in o['label']]
                dev_f1, dev_T2, precision, recall, best_pred, best_label = self.f1_metric(label, pred)
                dev_acc = self.acc_metric(label, pred)
            elif i == 1:
                pred = [oo  for o in output for oo in o['pred']]
                #dev_data = get_original_data(self.config.val_f)
                #_, _, dev_f1c, dev_T2c = self.f1c_metric(pred, dev_data)
            elif i == 2:
                test_loss = torch.stack([x['test_loss'] for x in output]).mean()
                pred = [oo  for o in output for oo in o['pred']]
                label = [oo for o in output for oo in o['label']]
                test_f1, _, precision, recall, best_pred, best_label = self.f1_metric(label, pred, T2=dev_T2)
                test_acc = self.acc_metric(label, pred)
            elif i == 3:
                pred = [oo  for o in output for oo in o['pred']]
                #tst_data = get_original_data(self.config.test_f)
                #_, _, tst_f1c, _ = self.f1c_metric(pred, tst_data, T2=dev_T2c)

        tensorboard_logs = {
            'dev_loss': dev_loss,
            'test_loss': test_loss, 
            'dev_acc': dev_acc,
            'test_acc': test_acc,
            # 'test_T2': test_T2,
            # 'precision': precision,
            # 'recall': recall,
        }
        
        with open(os.path.join(self.logger.log_dir, 'output'), 'wb') as f:
            pickle.dump([best_pred, best_label], f) 
        return {'progress_bar': tensorboard_logs, 'log': tensorboard_logs}

    def train_dataloader(self):
        kwargs = dict(num_workers=self.config.num_workers, batch_size=self.config.batch_size, collate_fn=collate_fn, pin_memory=True)
        return data.DataLoader(self.ds_trn, shuffle=True, **kwargs) 

    def val_dataloader(self):
        kwargs = dict(shuffle=False, num_workers=self.config.num_workers, batch_size=self.config.batch_size, collate_fn=collate_fn, pin_memory=True)
        return data.DataLoader(self.ds_val, **kwargs)

    def test_dataloader(self):
        kwargs = dict(shuffle=False, num_workers=self.config.num_workers, batch_size=self.config.batch_size, collate_fn=collate_fn, pin_memory=True)
        dl_val = data.DataLoader(self.ds_val, **kwargs)
        dl_val_c = data.DataLoader(self.ds_val_c, **kwargs)
        dl_tst = data.DataLoader(self.ds_tst, **kwargs)
        dl_tst_c = data.DataLoader(self.ds_tst_c, **kwargs)
        return [dl_val, dl_val_c, dl_tst, dl_tst_c]
    def softmax(self, x):
        '''(N,in_features)of x softmax'''
        x_exp=torch.exp(x)
        return x_exp/x_exp.sum(dim=-1,keepdim=True)        
    @property
    def batch_size(self): return self.hparams.batch_size

    @batch_size.setter
    def batch_size(self, batch_size): self.hparams.batch_size = batch_size

    @property
    def lr(self): return self.hparams.lr

    @lr.setter
    def lr(self, lr): self.hparams.lr = lr