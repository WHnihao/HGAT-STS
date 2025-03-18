import numpy as np
import torch
import torch
import torch.nn as nn
import dgl
import torch.nn.functional as F
from model.global_model import ConvEncoder
#from model.local_model import UtterEncoder
from utils.config import config

class Graph_STS(nn.Module):

    def __init__(self, config, vocab, index_epoch, vocab_ph, vocab_sent, embed_dic_sent, embed_dic_phrase, embed_dic_word, phrase_dic_dic):
        super().__init__()
        self.embed_dic_sent, = embed_dic_sent,
        self.embed_dic_phrase = embed_dic_phrase
        self.embed_dic_word = embed_dic_word
        self.phrase_dic_dic = phrase_dic_dic
        self.config = config
        self.vocab = vocab
        self.vocab_ph = vocab_ph
        self.vocab_sent = vocab_sent
        self.global_model = ConvEncoder(config, self.vocab, self.vocab_ph, self.vocab_sent, self.embed_dic_sent, self.embed_dic_phrase, self.embed_dic_word, self.phrase_dic_dic)
        self.normalization = torch.nn.GroupNorm(2, 32, eps=1e-05) #, affine=True
        self.normalization1 = torch.nn.BatchNorm1d(1, eps=1e-05, momentum=0.1, affine=True)
        self.full = nn.Linear(config.embed_dim, config.embed_dim)
        
        self.classifier_dropout = nn.Dropout(config.dropout_rate)
        self.activation = nn.Tanh()##nn.LeakyReLU()#nn.PReLU()#nn.Softsign()#nn.ReLU()#nn.ReLU() # nn.Tanh()
        self.w1 = nn.Linear(config.embed_dim * 1, config.embed_dim * 1)
        self.w2 = nn.Linear(config.embed_dim * 1, config.embed_dim * 1)
        self.w4 = nn.Linear(config.embed_dim * 1, config.embed_dim * 1)
        self.w3 = nn.Linear(config.embed_dim * 1, 1)
        self.w5 = nn.Linear(config.embed_dim * 1, 1)
        self.w6 = nn.Linear(config.embed_dim * 1, 1)
        self.w7 = nn.Linear(config.embed_dim * 1, 1)
        self.classifier = nn.Linear(config.embed_dim * 1, 3) # total 36 relations   总共36种关系

    def forward(self, batch):
        #graph = batch['batch_graph']
        graph1 = batch['batch_graph1']
        dict_id = batch['dict_id']
        rids = torch.argmax(batch['rids'], dim=1)
        x_node_id, y_node_id, p_node_id, pl_node_id, pr_node_id, plo_node_id, pro_node_id, x_node_id_full, y_node_id_full, p_node_id_full, plo_node_id_full, pro_node_id_full = batch['x_node_id'], batch['y_node_id'], batch['p_node_id'], batch['pl_node_id'], batch['pr_node_id'], batch['plo_node_id'], batch['pro_node_id'], batch['x_node_id_full'], batch['y_node_id_full'], batch['p_node_id_full'], batch['plo_node_id_full'], batch['pro_node_id_full']  

        self.global_model(graph1, dict_id, batch)  #, graph
        
        arga, argb, arglo_full, argro_full, argc, reg_loss = self.unbatch_graph(graph1, x_node_id, y_node_id, p_node_id, pl_node_id, pr_node_id, plo_node_id, pro_node_id, x_node_id_full, y_node_id_full, p_node_id_full, plo_node_id_full, pro_node_id_full, rids) # (batch, embed_dim)  , graph
        reg_loss = reg_loss.mean()#.sum(dim=0)   , argl, argr, arglo, argro

        
        
        y = torch.mul(arga, argb)
        x = torch.abs(torch.sub(arga, argb))
        
        
        
 
        y = self.w1(x) + self.w2(y)
        y = self.activation(y)
        y = self.classifier_dropout(y)
        y = self.classifier(y)

        return y, argc, reg_loss
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
    def unbatch_graph(self, graph1,x_node_id, y_node_id, p_node_id, pl_node_id, pr_node_id, plo_node_id, pro_node_id, x_node_id_full, y_node_id_full, p_node_id_full, plo_node_id_full, pro_node_id_full, rids):    #, graph

        arga, argb, argc, argl, argr, arglo, argro, arglo_full, argro_full = [], [], [], [], [], [], [], [], []

        reg_loss = []
        for x_id, y_id, p_id, pl_id, pr_id, plo_id, pro_id, x_id_full, y_id_full, p_id_full,  plo_id_full, pro_id_full, rid in zip(x_node_id, y_node_id, p_node_id, pl_node_id, pr_node_id, plo_node_id, pro_node_id, x_node_id_full, y_node_id_full, p_node_id_full, plo_node_id_full, pro_node_id_full, rids):    
            x_feat = graph1.nodes[x_id_full].data['feat'] # (number_of_x_word, embed_dim)
            #p_feat = graph1.nodes[p_id_full].data['feat']
            #print(plo_id_full)
            #print(pro_id_full)
            plo_full_feat = graph1.nodes[plo_id_full].data['feat'] 
            pro_full_feat = graph1.nodes[pro_id_full].data['feat']

            pro_full_feat1 = torch.transpose(pro_full_feat, 0, 1)

            pro_full_feat_ten = self.activation(pro_full_feat1)
            plo_full_feat_ten = self.activation(plo_full_feat)
            plo_full_feat_at1 = torch.matmul(plo_full_feat_ten, pro_full_feat1)

            plo_full_feat_at = self.w6(plo_full_feat)
            pro_full_feat_at = self.w7(pro_full_feat)


            plo_full_feat_at = nn.functional.softmax(plo_full_feat_at.squeeze(-1))

            pro_full_feat_at = nn.functional.softmax(pro_full_feat_at.squeeze(-1))

            plo_full_feat = (plo_full_feat_at.unsqueeze(-1) * plo_full_feat).sum(dim=0)
            pro_full_feat = (pro_full_feat_at.unsqueeze(-1) * pro_full_feat).sum(dim=0)

            reg_loss_l = plo_full_feat_at.unsqueeze(-1).pow(2).mean()#.sum(dim=0)#.mean()
            reg_loss_r = pro_full_feat_at.unsqueeze(-1).pow(2).mean()#.sum(dim=0)

            reg_loss.append((reg_loss_l + reg_loss_r)/2)

            y_feat = graph1.nodes[y_id_full].data['feat']

            x_feat, _ = torch.max(x_feat, dim=0) 

            y_feat, _ = torch.max(y_feat, dim=0)

            arglo_full.append(plo_full_feat)
            argro_full.append(pro_full_feat)
            arga.append(x_feat)
            argb.append(y_feat)

            argc.append(y_feat)

        return torch.stack(arga), torch.stack(argb), torch.stack(arglo_full), torch.stack(argro_full), torch.cat(argc, dim=0), torch.stack(reg_loss)     

