import torch
import torch.utils.data as data
import random, os, re, ast, time
import numpy as np
import collections
import dgl
import torch.nn as nn
from dgl.data.utils import save_graphs, load_graphs
from nltk.corpus import stopwords

from utils.torch_utils import get_positions
from utils.config import config
from utils import constant
from transformers import BertTokenizer, BertModel, BertForMaskedLM # BertModel, BertConfig, BertTokenizer
stopwords = stopwords.words('english')

#from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
# Load pretrained model/tokenizer

class Bertsent(nn.Module):
    def __init__(self, pooling):
        super().__init__()
        self.pooling = pooling
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  #bert-base-uncased
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.w = nn.Linear(config.embed_dim, config.embed_dim)
        # modelConfig = BertConfig.from_pretrained('bert-base-chinese-config.json')
        # self.textExtractor = BertModel.from_pretrained('bert-base-chinese-pytorch_model.bin', config=modelConfig)
        # self.textExtractor = BertModel.from_pretrained('bert-base-chinese-pytorch_model.bin', config=modelConfig)
    def forward(self, text):
        tokens, segments, input_masks = [], [], []
        for sent in text:
            #tokenized_text = tokenizer.tokenize(text) #用tokenizer对句子分词
            #####print(sent)
            sent_num = self.tokenizer(sent)
            #sent_nophrase_num_list.append(sent_num['input_ids'])
            #sent_nophrase_num1 = self.tokenizer(sent_nophrase_nolist[1])
            #indexed_tokens = self.tokenizer.convert_tokens_to_ids(sent)#索引列表
            tokens.append(sent_num['input_ids'])
            ####print("111111111111111111111111111111111", tokens, len(sent_num['input_ids']), len(sent_num['token_type_ids']), len(sent_num['attention_mask']))
            segments.append(sent_num['token_type_ids'])
            input_masks.append(sent_num['attention_mask'])

        max_len = max([len(single) for single in tokens]) #最大的句子长度
        #print(max_len)
        for j in range(len(tokens)):
            padding = [0] * (max_len - len(tokens[j]))
            tokens[j] += padding
            segments[j] += padding
            input_masks[j] += padding
        tokens = torch.tensor(tokens, dtype = torch.int64)
        #print(len(input_masks)), device=config.device, device=config.device, device=config.device
        #print(tokens)
        input_masks = torch.tensor(input_masks, dtype = torch.int64)
        segments = torch.tensor(segments, dtype = torch.int64)
        out = self.bert(tokens)
        # return out[1]
        if self.pooling == 'cls':
            #out = out.last_hidden_state[:, 0]
            #out = self.w(out)
            return out.last_hidden_state[:, 0]  # [batch, 768]
        if self.pooling == 'pooler':
            return out.pooler_output  # [batch, 768]
        if self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)  # [batch, 768, seqlen]
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1) 
# class BertTextNet(nn.Module):
    # def __init__(self,  code_length): #code_length为fc映射到的维度大小
        # super(BertTextNet, self).__init__()

        # modelConfig = BertConfig.from_pretrained('bert-base-chinese-config.json')
        # self.textExtractor = BertModel.from_pretrained(
            # 'bert-base-chinese-pytorch_model.bin', config=modelConfig)
        # embedding_dim = self.textExtractor.config.hidden_size

        # self.fc = nn.Linear(embedding_dim, code_length)
        # self.tanh = torch.nn.Tanh()

    # def forward(self, tokens, segments, input_masks):
        ##output=self.textExtractor(tokens, token_type_ids=segments, attention_mask=input_masks)

