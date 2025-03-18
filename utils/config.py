import torch
import os
import sys
import argparse
from argparse import Namespace
import numpy as np
import random


parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='train', help='train | test')
parser.add_argument('--data_dir', type=str, default='dataset/')
parser.add_argument('--glove_file', type=str, default='glove.6B.300d.txt')
parser.add_argument('--ckpt_path', type=str, default='')
parser.add_argument('--comment', type=str, default='')

parser.add_argument('--embed_dim', type=int, default=300, help='Word embedding dimension.,')
parser.add_argument('--ner_embed_dim', type=int, default=30, help='NER embedding dimension. concat with word embedding on dim2')
parser.add_argument('--pos_embed_dim', type=int, default=30, help='POS embedding dimension. concat with word embedding on dim2')
parser.add_argument('--lgcn_hidden_dim', type=int, default=300, help='Local GCN hidden size.')
parser.add_argument('--input_dropout', type=float, default=0.1, help='Input dropout rate for word embeddings')
parser.add_argument('--min_freq', type=int, default=0, help='min frequency')
parser.add_argument('--tune_topk', type=int, default=1e10, help='Only finetune top N word embeddings.')
parser.add_argument('--lower', dest='lower', action='store_true', default=True, help='Lowercase all words.')

parser.add_argument('--pool_type', choices=['max', 'avg', 'sum'], default='max', help='Pooling function type for local gcn. Default max.')

parser.add_argument('--rnn_hidden_dim', type=int, default=300, help='RNN hidden state size.')
parser.add_argument('--rnn_layers', type=int, default=1, help='Num of RNN layers.')
parser.add_argument('--rnn_dropout', type=float, default=0.1, help='RNN dropout rate.')

parser.add_argument('--lr', type=float, default=3e-4, help='learning rate initial 5e-4')
parser.add_argument('--max_lr', type=float, default=1e-3, help='maximum learning rate for cyclic learning rate')
parser.add_argument('--base_lr', type=float, default=5e-5, help='minimum learning rate for cyclic learning rate')
parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam', 'adamw', 'adamax'], default='adam', help='Optimizer: sgd, adamw, adamax, adam')
parser.add_argument('--scheduler', type=str, choices=['', 'exp', 'cyclic'], default='exp', help='use scheduler')
parser.add_argument('--lr_decay', type=float, default=0.87, help='scheduler decay')
parser.add_argument('--num_epoch', type=int, default=20, help='Number of total training epochs.')
parser.add_argument('--batch_size', type=int, default=32, help='batch size cuda can support')
parser.add_argument('--actual_batch_size', type=int, default=32, help='actual batch size that you want')
parser.add_argument('--save_dir', type=str, default='lightning_logs', help='Root dir for saving models.')

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--model_file', type=str, help='Filename of the pretrained model.')

parser.add_argument('--glstm_hidden_dim', type=int, default=300, help='size of global lstm hidden state [default: 128]')
parser.add_argument('--glstm_layers', type=int, default=2, help='Number of global lstm layers [default: 2]')
parser.add_argument('--glstm_dropout_prob', type=float, default=0.2,help='recurrent dropout prob [default: 0.1]')

parser.add_argument('--ggcn_n_iter', type=int, default=1, help='iteration hop [default: 1] for global GCN')
parser.add_argument('--ggcn_layers', type=int, default=1, help='Number of GAT layers [default: 1]')
parser.add_argument('--ggcn_hidden_size', type=int, default=300, help='final output size & sentence node hidden size [default: 300]')
parser.add_argument('--edge_embed_size', type=int, default=50, help='feature embedding size for edge[default: 50]')
parser.add_argument('--ffn_inner_hidden_size', type=int, default=300,help='PositionwiseFeedForward inner hidden size [default: 512]')
parser.add_argument('--word2sent_n_head', type=int, default=10, help='multihead attention number [default: 10]')
parser.add_argument('--sent2word_n_head', type=int, default=10, help='multihead attention number [default: 10]')
parser.add_argument('--atten_dropout_prob', type=float, default=0.2, help='attention dropout prob [default: 0.1]')
parser.add_argument('--ffn_dropout_prob', type=float, default=0.2,help='PositionwiseFeedForward dropout prob [default: 0.1]')


parser.add_argument('--dropout_rate', type=float, default=0.5, help="dropout rate for classifier")

parser.add_argument('--rm_stopwords', type=bool, default=False, help='Remove stopwords in global word Node')

args = parser.parse_args()



class Config:
    def __init__(self, args):
        self.__dict__.update(vars(args))
        self.gcn_lin_dim = self.lgcn_hidden_dim
        self.ws_edge_bucket = 10
        self.wn_edge_bucket = 10
        self.train1_f, self.train2_f, self.train3_f, self.train4_f, self.train5_f, self.train6_f, self.val_f, self.test_f = (os.path.join(self.data_dir, o) for o in ['snli_train1.txt', 'snli_train2.txt', 'snli_train3.txt','snli_train4.txt', 'snli_train5.txt', 'snli_train6.txt','snli_test.txt', 'snli_dev.txt']) #When the corpus is too large, we opt to process the training data in batches.
        #
        self.glove_f = os.path.join(self.data_dir, self.glove_file)
        self.embed_f = os.path.join(self.data_dir, 'embeddings.npy')        
        self.proce_f = os.path.join(self.data_dir, 'dataset_preproc.p')
        self.proce_f_sentence = os.path.join(self.data_dir, 'dataset_preproc_sentence_128.p')
        self.proce_f_phrase = os.path.join(self.data_dir, 'dataset_preproc_phrase_128.p')
        self.proce_f_word = os.path.join(self.data_dir, 'dataset_preproc_word_128.p')
        self.proce_f_c = os.path.join(self.data_dir, 'dataset_preproc_c.p')
        self.w_p_s_list = os.path.join(self.data_dir, 'W_p_s.p')
        self.num_workers = self.batch_size//2 
        self.gpus = 1
        #self.n_gpu = torch.cuda.device_count()
        #self.device = torch.device('cuda:0' if self.n_gpu > 0 else 'cpu')
        #self.device_ids = list(range(self.n_gpu))
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.num_nodes = 1
        self.precision = 32
        if self.mode == 'test': assert len(self.ckpt_path) > 0, "Please provide a --ckpt_path"

config = Config(args)
