import torch
import torch.utils.data as data
import random, os, re, ast, time
import numpy as np
import collections
import dgl
from dgl.data.utils import save_graphs, load_graphs
from nltk.corpus import stopwords

from utils.torch_utils import get_positions
from utils.config import config
from utils import constant

stopwords = stopwords.words('english')
#from transformers import BertTokenizer, BertModel, BertForMaskedLM # BertModel, BertConfig, BertTokenizer
#from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
# Load pretrained model/tokenizer

class Dataset(data.Dataset):
    def __init__(self, data, vocab, phrase_list_repet, phrase_list_pos_repet, vocab_ph, vocab_sent, v_list, s_list, p_list):
        self.vocab = vocab
        self.data = data 
        self.phrase_list_repet = phrase_list_repet
        self.phrase_list_pos_repet = phrase_list_pos_repet
        self.vocab_ph = vocab_ph#vocab_ph, vocab_sent, v_list, s_list, p_list
        self.vocab_sent = vocab_sent
        self.v_list = v_list
        self.s_list = s_list
        self.p_list = p_list

    def __len__(self):
        return len(self.data)


    def conv_to_ids(self, conv):
        conv_ids = [torch.LongTensor(self.vocab.map(o)) for o in conv]
        return conv_ids
    def conv_to_ids_list(self, conv):
        conv_ids = [self.vocab.map(o) for o in conv]
        return conv_ids
    def phrase_to_ids(self, conv):
        conv_ids = [torch.LongTensor(self.vocab_ph.map(o)) for o in conv]
        return conv_ids
    def sent_to_ids(self, conv):
        conv_ids = [torch.LongTensor(self.vocab_sent.map(o)) for o in conv]
        return conv_ids
    def phrase_to_ids_list(self, conv):
        conv_ids = [self.vocab_ph.map(o) for o in conv]
        return conv_ids
    def sent_to_ids_list(self, conv):
        conv_ids = [self.vocab_sent.map(o) for o in conv]
        return conv_ids

    def label_to_oneHot(self, rel_labels):
        rid = [0, 0, 0]
        labels_list = ['neutral', 'contradiction', 'entailment']
        #rids = [4]
        #ind_lable = 0
        #print("rel_labels", rel_labels)
        for ind, i in enumerate(labels_list):
            if i ==rel_labels:
                 rid[ind] = 1
        return torch.FloatTensor(rid)


    def remove_stopwords(self, utter): return [w for w in utter if w not in stopwords]

        
        
    def add_wordNode(self, G, num_nodes, word_ids, x_wids=None, y_wids=None):
        """word: unit=0, dtype=0
        """
        G.add_nodes(num_nodes)
        wid2nid = {w:i for i,w in enumerate(word_ids)}    
        nid2wid = {i:w for w,i in wid2nid.items()}#序号对应单词
        G.ndata['unit'] = torch.zeros(num_nodes)  
        G.ndata['dtype'] = torch.zeros(num_nodes)    
        G.ndata['id'] = torch.LongTensor(word_ids) 
        return wid2nid, nid2wid
    def add_phraseNode(self, G, num_nodes, start_ids, phrase_ids_list):
        """sent: unit=1, dtype=2"""
        G.add_nodes(num_nodes)
        pid2nid = {w: i+start_ids for i,w in enumerate(phrase_ids_list)}#是在词节点的基础上进行编号
        nid2pid = {i+start_ids: w for i,w in enumerate(phrase_ids_list)}  #和上面一条一样只是把序列和
        G.ndata['unit'][start_ids:] = torch.ones(num_nodes)   #节点的类型定义一下
        G.ndata['dtype'][start_ids:] = torch.ones(num_nodes)  #类型节点也是从上一节点编号结束开始
        #wordids = list(pid2nid.keys(), key=lambda o: pid2nid[o])     #利用单词词频字典，对单词进行频次排序
        
        G.ndata['id'][start_ids:] = torch.LongTensor(phrase_ids_list)
        return pid2nid, nid2pid
    def add_sentNode(self, G, num_nodes, start_ids, senid):
        """sent: unit=1, dtype=3"""
        G.add_nodes(num_nodes)
        
        sid2nid = {w: i+start_ids for i,w in enumerate(senid)}
        nid2sid = {i+start_ids: w for i,w in enumerate(senid)}  
        G.ndata['unit'][start_ids:] = torch.ones(num_nodes)* 2
        G.ndata['dtype'][start_ids:] = torch.ones(num_nodes) * 2
        G.ndata['id'][start_ids:] = torch.LongTensor(senid)
        return sid2nid, nid2sid

    def add_speakerNode(self, G, num_nodes, start_ids):#看一下speakerNode节点的特点
        """speaker: unit=0, dtype=3"""
        G.add_nodes(num_nodes)
        speakerid2nid = {i: i+start_ids for i in range(num_nodes)}
        nid2speakerid = {i+start_ids: i for i in range(num_nodes)}
        G.ndata['unit'][start_ids:] = torch.zeros(num_nodes)
        G.ndata['dtype'][start_ids:] = torch.ones(num_nodes) * 3
        G.ndata['id'][start_ids:] = torch.arange(num_nodes)
        return speakerid2nid, nid2speakerid



    def get_weight(self): 
        return torch.randint(config.ws_edge_bucket, (1,) )
    def creat_graph(self, utters, phrase_before, phrase_after, phrase_tag, utters_phrase, phrase_word, tokens_phrase, utters_words, utters_tags,  tokens_word, tokens_pos, words2tags, tags2words, sent_id, word_el, sent_nophrase, sent_before, phrasenosent_after, sent_after, phrasenosent_before, label, phrase_pos1, phrase_tags1, sent_pos, pos_tags, phrase_tags, phrase_pos):
        """create graph for each sentence tree
      

        graph: dgl.DGLGraph
            node:
                word : unit=0, dtype=0      # 单词三种节点的结构不一致
                sent: unit=1, dtype=1       # 句子
                phrase: unit=1, dtype=2        # 短语

            edge:
                word & phrase - sent: dtype=0   #单词&短语 - 句子
                word & phrase - phrase: dtype=1              #单词&短语 - 短语
        """
        G = dgl.graph([])
        G.set_n_initializer(dgl.init.zero_initializer)

        phrase_NP, phrase_VP, phrase_NP_s, phrase_VP_s = [], [], [], []
        #print("123321", sid2nid)
        for int_sent, sent in enumerate(utters):
            for ind, phrase in enumerate(phrase_before[int_sent]):
                #print("123321", phrase_tags1[0][ind])
                if sent == phrase:
                    if phrase_tags1[int_sent][ind] == 'VP' and len(phrase_after[int_sent][ind]) > 1 :
                        #print(phrase_after[int_sent][ind])
                        #print(phrase_after[int_sent][ind])
                        phrase_VP.append([int_sent, ind])
                        for ind_be, before in enumerate(phrase_before[int_sent]):
                            if before == phrase_after[int_sent][ind_be]:
                                #print(int_sent, phrase_after[int_sent][ind_be])
                                if phrase_tags1[int_sent][ind_be] == 'VP':#and len(phrase_after[int_sent][ind_be]) >1 :
                                    phrase_VP_s.append([int_sent, ind_be])
                                    #print(int_sent, phrase_after[int_sent][ind_be])
                    if phrase_tags1[int_sent][ind] == 'NP' and len(phrase_after[int_sent][ind]) > 1:
                        phrase_NP.append([int_sent, ind])
                        for ind_be, before in enumerate(phrase_before[int_sent]):
                            if before == phrase_after[int_sent][ind_be]:
                                #print(int_sent, phrase_after[int_sent][ind_be])
                                if phrase_tags1[int_sent][ind_be] == 'NP':#and len(phrase_after[int_sent][ind_be]) >1 :
                                    phrase_VP_s.append([int_sent, ind_be])

        phrase_list_random, phrase_list_pos_random = [], []
        nums = random.sample(range(0,len(self.phrase_list_repet)), 10)

        for i in nums:
            phrase_list_random.append(self.phrase_list_repet[i])
            phrase_list_pos_random.append(self.phrase_list_pos_repet[i])
        utters_ids = self.conv_to_ids(sent_nophrase)         #又一次进行单词的数字化表示,utters应该是一系列的单词在列表中, utters是所有的句子集合
        #print(utters_ids)
        ####print("16181618161816181618161816181618", phrase_word)
        phrase_pos_random = [ [constant.pos_s2i[oo] for oo in o] for o in phrase_list_pos_random ]
        phrase_word_1 = phrase_word + phrase_list_random
        phrase_pos_1 = phrase_pos + phrase_pos_random
        phrase_word_merge, phrase_pos_merge = [], []
        phrase_word_merge = phrase_word
        #print(phrase_word_merge)
        phrase_pos_merge = phrase_pos
        
        utters_phrase_ids_random = self.conv_to_ids(phrase_word_merge)
        utters_phrase_ids = self.conv_to_ids(phrase_word)          #属于所有单词和短语进行数字化，这个需不需要和短语一起数字化
        #print("11221", pos_tags, phrase_tags)
        #print(utters)
        if config.rm_stopwords:           #去除停用词
            words = [w for u in utters for w in self.remove_stopwords(u)]
        else:
            words = word_el #sent_nophrase_num_list[0] + sent_nophrase_num_list[1]#[w 
        words_ids = sorted(list(set(self.vocab.map(words))))
        #words_ids = list(self.vocab.map(words))  #sorted(   #单词映射到数字并且进行排序，words包含句子中所有的单词\，单词映射到
        #phrases_ids = sorted(list(set(self.vocab.map(phrases))))
        #wid2nid, nid2wid = self.add_wordNode(G, len(words_ids), words_ids)   #节点我们包含三种节点句子节点，短语节点，单词节点
        #print(words_ids)
        phrase_before_ind, phrase_after_ind, sent_before_ind, sent_after_ind, phrase_random, phrase_word_only, phrase_before_only_ind  = [], [], [], [], [], [], []

        phrase_word_merge_ids = self.phrase_to_ids_list([phrase_word_merge])
        #print("11111", phrase_word_merge_ids)
        phrase_word_merge_ids = phrase_word_merge_ids[0]
        #print("22222", phrase_word_merge_ids)
        phrase_word_merge_unmap_ids = self.vocab_ph.unmap(phrase_word_merge_ids)
        #print("1111111111", phrase_word_merge_unmap_ids)
        # word_el_nolist = ""
        # for word in word_el:
            # word_el_nolist = word_el_nolist + " " + word
        # word_ids_list_dic = self.tokenizer(word_el_nolist)
        # word_ids_list = word_ids_list_dic["input_ids"]
        sentence_ids_list = self.sent_to_ids_list([sent_nophrase])
        #print("11111", phrase_word_merge_ids)
        sentence_ids_list = sentence_ids_list[0]
        #print(word_ids_list)
        word_ids_all = self.conv_to_ids_list([words])
        word_ids_list = word_ids_all[0]
        word_ids_list_de = []
        low_frequency = []
        low_locate = []
        if len(word_ids_list) == len(words_ids):
            word_ids_list_de = word_ids_list
        else:
            for ind, word_id in enumerate(word_ids_list):
                if word_id not in word_ids_list_de:
                    word_ids_list_de.append(word_id)
                else:
                    for ind_low, word_low in enumerate(word_ids_list):
                        if word_id == word_low:
                            low_frequency.append(ind) 
                            low_locate.append(word_low)#前面ind是重复低频词的位置， 

        for i, phrase_rad in enumerate(phrase_list_random):
            #print("before_0", i, len(sent_before), len(phrase_before))
            #phrase_random.append([])
            
            for ind_phrase, phrase_list in enumerate(phrase_word_merge):
                    #print("phrase_list", phrase_list)
                    if phrase_rad == phrase_list:
                        phrase_random.append([0, phrase_word_merge_ids[ind_phrase]])
        for i, phrase_w in enumerate(phrase_word):
            #print("before_0", i, len(sent_before), len(phrase_before))
            #phrase_word_only.append([])
            
            for ind_phrase, phrase_list in enumerate(phrase_word_merge):
                    #print("phrase_list", phrase_list)
                    if phrase_w == phrase_list:
                        phrase_word_only.append([0, phrase_word_merge_ids[ind_phrase]])
        for i, sent_be in enumerate(phrase_before):
            #print("before_0", i, len(sent_before), len(phrase_before))
            phrase_before_ind.append([])
            for ind, before in enumerate(sent_be): 
                a = 0
                #print("121212121212121212121212", phrase_word)
                for ind_phrase, phrase_list in enumerate(phrase_word_merge):
                    #print("phrase_list", phrase_list)
                    
                    if before == phrase_list:
                        phrase_before_ind[-1].append([0, phrase_word_merge_ids[ind_phrase]])
                        a = 1
                        # for ind_sent, sent_list in enumerate(sent_nophrase):
                            # if before == sent_list:
                                # print("1111111111111111111")
                        #print("0before, phrase_list", before, phrase_list)
                #print("phrase_before_ind[-1]", len(phrase_before_ind[-1]))
                b = 0
                if a == 0:
                   
                    for ind_word, word_list in enumerate(word_el):
                        if before[0] == word_list and len(before) < 2: 
                            if ind_word not in low_frequency:
                                b = 1
                                #print("1before, word_list", before, word_list, ind_word)
                                phrase_before_ind[-1].append([1, word_ids_list[ind_word]])
                            else:
                                b = 1
                                #print("1before, word_list", before, word_list, ind_word)
                                phrase_before_ind[-1].append([1, low_locate[0]])
                if a == 0 and b == 0:
                    for ind_sent, sent_list in enumerate(sent_nophrase):
                        if before == sent_list:
                            b = 1
                            #print("1before, sent_list", before, sent_list, ind_sent)
                            phrase_before_ind[-1].append([2, sentence_ids_list[ind_sent]])
        

        phrase_before_pairing = []
        for i_be, sent_be in enumerate(phrase_before): 
            #print("before_0", i, len(sent_before), len(phrase_before))
            phrase_before_only_ind.append([])
            only = []
            for ind, before in enumerate(sent_be): 
                a = 0
                
                for ind_phrase, phrase_list in enumerate(phrase_word_merge):
                    
                    if before == phrase_list:
                        a = 1
                    if before == phrase_list and phrase_list not in only:
                        only.append(before)
                        phrase_before_only_ind[-1].append([0, phrase_word_merge_ids[ind_phrase]])
                        
                       
                b = 0
                if a == 0:
                    

                    for  ind_word, word_list in enumerate(word_el):
                        if before[0] == word_list and len(before) < 2:
                            if ind_word not in low_frequency:
                                b = 1
                                only.append(before)
                                #print("1before, word_list", before, word_list)
                                phrase_before_only_ind[-1].append([1, word_ids_list[ind_word]])
                            else:
                                b = 1
                                only.append(before)
                                #print("1before, word_list", before, word_list)
                                phrase_before_only_ind[-1].append([1, low_locate[0]])
                if a == 0 and b == 0:
                    for  ind_sent, sent_list in enumerate(sent_nophrase):
                        if before == sent_list and before not in only:
                            b = 1
                            only.append(before)
                            #print("1before, word_list", before, word_list)
                            phrase_before_only_ind[-1].append([2, sentence_ids_list[ind_sent]])
        for i_be, sent_be in enumerate(phrase_before): #两个句子，分开来陈列
            #print("before_0", i, len(sent_before), len(phrase_before))
            #phrase_before_ind.append([])
            ind_before = []
            phrase_before_trunk = []
            
            for ind_be0, before_0 in enumerate(sent_be): #每个句子分别在遍历里面的句子
                ind_leaf = []
                phrase_before_0 = []
                a = 0
                if ind_be0 not in ind_before:
                    for ind_phrase, phrase_list in enumerate(phrase_word_merge):
                        if before_0 == phrase_list:
                            phrase_before_0.append([0, phrase_word_merge_ids[ind_phrase]])
                            a = 1
                        #print("0before, phrase_list", before, phrase_list)
                #print("phrase_before_ind[-1]", len(phrase_before_ind[-1]))

                    if a == 0:
                        b = 0
                        for  ind_word, word_list in enumerate(word_el):
                            if before_0[0] == word_list and len(before_0) < 2:
                                if ind_word not in low_frequency:
                                    b = 1
                                    #print("1before, word_list", before, word_list)
                                    phrase_before_0.append([1, word_ids_list[ind_word]])
                                else:
                                    b = 1
                                    #print("1before, word_list", before, word_list)
                                    phrase_before_0.append([1, low_locate[0]])
                    
                    for ind_be1, before_1 in enumerate(sent_be):
                        if before_0 == before_1:
                            #ind_before.append()
                            ind_before.append(ind_be1)
                            for i in range(ind_be0, len(sent_be)):
                                if phrase_after[i_be][ind_be0] == sent_be[i]:
                                    ind_leaf.append(i)
                    #print(phrase_before_0)
                    if b == 1:
                        phrase_before_trunk.append([])
                        phrase_before_trunk[-1].append(phrase_before_0[0])
                        phrase_before_trunk[-1].append(ind_leaf)
            phrase_before_pairing.append(phrase_before_trunk)
        for i, sent in enumerate(sent_before):
            #print("before_0", i, len(sent_before), len(phrase_before))
            sent_before_ind.append([])
            for ind, before in enumerate(sent): 
                a = 0
                #print("before_0", before)
                for ind_phrase, phrase_list in enumerate(sent_nophrase):
                    #print("phrase_list", phrase_list)
                    if before == phrase_list:
                        sent_before_ind[-1].append([0, sentence_ids_list[ind_phrase]])
                        a = 1
                        #print("0before, phrase_list", before, phrase_list)
                #print("phrase_before_ind[-1]", len(phrase_before_ind[-1]))
                if a == 0:
                    b = 0

                    for  ind_word, word_list in enumerate(word_el):
                        if before[0] == word_list:
                            if ind_word not in low_frequency:
                                b = 1
                                #print("1before, word_list", before, word_list)
                                sent_before_ind[-1].append([1, word_ids_list[ind_word]])
                            else:
                                b = 1
                                #print("1before, word_list", before, word_list)
                                sent_before_ind[-1].append([1, low_locate[0]])

        for i, sent_be in enumerate(phrase_after):
            #print("before_1", i, len(phrase_after[i]), len(phrase_after))
            phrase_after_ind.append([])
            for ind, after in enumerate(phrase_after[i]):
                a = 0

                for ind_phrase, phrase_list in enumerate(phrase_word_merge):

                    if after == phrase_list:
                        phrase_after_ind[-1].append([0, phrase_word_merge_ids[ind_phrase]])
                        a = 1


                if a == 0:
                    b =0
                    for  ind_word, word_list in enumerate(word_el):
                        #print(phrase_word)
                        #print("words", after, words)
                        if after[0] == word_list:
                            if ind_word not in low_frequency:
                                #print("1after, word_list", after, word_list)
                                b = 1
                                phrase_after_ind[-1].append([1, word_ids_list[ind_word]])
                            else:
                                b = 1
                                phrase_after_ind[-1].append([1, low_locate[0]])
        #print("123454321", sent_after, "123321", sent_before)
        for i, sent_be in enumerate(sent_after):
           ##### print("before_1", i, len(phrase_after[i]), len(phrase_after))
            sent_after_ind.append([])
            for ind, after in enumerate(sent_after[i]):
                a = 0
                for ind_phrase, phrase_list in enumerate(phrase_word_merge):
                    if after == phrase_list:
                        sent_after_ind[-1].append([0, phrase_word_merge_ids[ind_phrase]])
                        a = 1

                        #print("0after, phrase_list", after, phrase_list)
                #print("phrase_after_ind[-1]", len(phrase_after_ind[-1]))
                if a == 0:                     #a=0说明sent_after[]是一个单词，不是短语
                    b =0
                    #print("12344321", sent_after)
                    #print("54321", after)
                    #print("121212121111112121211121121", after)
                    for ind_word, word_list in enumerate(word_el):
                        #print(phrase_word)
                        #print("words", after, words)
                        if after[0] == word_list:
                            if ind_word not in low_frequency:
                                #print("1after, word_list", after, word_list)
                                b = 1
                                sent_after_ind[-1].append([1, word_ids_list[ind_word]])
                            else:
                                b = 1
                                sent_after_ind[-1].append([1, low_locate[0]])

        wordUtter_pos_dict = {}
        wordphrase_pos_dict = {}
        wordPhrase_pos_dict = {}
        PhrasePhrase_pos_dict = {} 
        for u_i, (w_ids, pos_ids) in enumerate(zip(utters_phrase_ids_random, phrase_pos_merge)):
            for ind_w, (w_i, pos_i) in enumerate(zip(w_ids, pos_ids)):
                for ind_id, word in enumerate(word_el):
                    if word == phrase_word_merge[u_i][ind_w]:
                        wordPhrase_pos_dict[(u_i, int(ind_id))] = torch.tensor([pos_i])      
        for u_i, (w_ids, pos_ids) in enumerate(zip(utters_ids, sent_pos)):
            for ind_w, (w_i, pos_i) in enumerate(zip(w_ids, pos_ids)): 
                for ind_id, word in enumerate(word_el):
                    if word == sent_nophrase[u_i][ind_w]:
                        wordUtter_pos_dict[(u_i, int(ind_id))] = torch.tensor([pos_i])   
        #print("1234", words_ids)  

        wid2nid, nid2wid = self.add_wordNode(G, len(word_ids_list_de), word_ids_list_de)
        pid2nid, nid2pid = self.add_phraseNode(G, len(phrase_word_merge_ids), len(word_ids_list_de), phrase_word_merge_ids)
        ####print("pid2nid", pid2nid)
        sid2nid, nid2sid = self.add_sentNode(G, len(sentence_ids_list) , len(word_ids_list_de) + len(phrase_word_merge_ids), sentence_ids_list)
        
        for ind_sent, sent_before in enumerate(phrase_before_pairing):
            for ind, phrase_id in enumerate(sent_before):
                if phrase_id[0][0] == 0:
                    
                    for ind_hou, phrase_id_hou in enumerate(sent_before):
                        if phrase_id_hou[0][0] ==0 and ind != ind_hou:
                            G.add_edges(pid2nid[phrase_id[0][1]], pid2nid[phrase_id_hou[0][1]], data={'ws_link': torch.tensor([phrase_tags[ind_sent][ind]]), "dtype": torch.tensor([5])})
        for sid in range(len(sent_nophrase)):       #range(len(utters_phrase_ids)):
            for ind_w, wid in enumerate(word_el):
                s_ids = sent_nophrase[sid]
                if wid in s_ids:# and sid in sent_id:
                    #print(sent_id)
                    #print(sent_id.index(sid))
                    ws_link = wordUtter_pos_dict[(sid, ind_w)]#wordUtter_pos_dict[(sent_id.index(sid), wid)]
                    G.add_edges(wid2nid[word_ids_list[ind_w]], sid2nid[sentence_ids_list[sid]], data={'ws_link': ws_link, "dtype": torch.tensor([2])})   #类型是
                    G.add_edges(sid2nid[sentence_ids_list[sid]], wid2nid[word_ids_list[ind_w]], data={'ws_link': ws_link, "dtype": torch.tensor([2])})
        for sid in range(len(phrase_word_merge)):       #range(len(utters_phrase_ids)):
            #if sid not in sent_id:
        #for sid in sent_id:       #range(len(utters_phrase_ids)):
            for ind_w, wid in enumerate(word_el):
                p_ids = phrase_word_merge[sid]
                if wid in p_ids:# and sid in sent_id:
                    #print(sent_id)
                    #print(sent_id.index(sid))
                    #print(sid, wid)
                    ws_link = wordPhrase_pos_dict[(sid, ind_w)]
                    G.add_edges(wid2nid[word_ids_list[ind_w]], pid2nid[phrase_word_merge_ids[sid]], data={'ws_link': ws_link, "dtype": torch.tensor([8])})   #类型是
                    G.add_edges(pid2nid[phrase_word_merge_ids[sid]], wid2nid[word_ids_list[ind_w]], data={'ws_link': ws_link, "dtype": torch.tensor([0])})

        dict_pid = {}
        for ind_sent, sent_before in enumerate(phrase_before_ind):
            for ind,  phrase_id in enumerate(sent_before):
                if phrase_id[0] == 0 and phrase_after_ind[ind_sent][ind][0] == 0:
                    G.add_edges(pid2nid[phrase_id[1]], pid2nid[phrase_after_ind[ind_sent][ind][1]], data={'ws_link': torch.tensor([phrase_tags[ind_sent][ind]]), "dtype": torch.tensor([1])})   
                    if dict_pid.get(pid2nid[phrase_id[1]], 0) == 0:
                        dict_pid[pid2nid[phrase_id[1]]] = [pid2nid[phrase_after_ind[ind_sent][ind][1]]]
                    #.append(pid2nid[phrase_id[1]])
                    else:
                        dict_pid[pid2nid[phrase_id[1]]].append(pid2nid[phrase_after_ind[ind_sent][ind][1]])
                    G.add_edges(pid2nid[phrase_after_ind[ind_sent][ind][1]], pid2nid[phrase_id[1]], data={'ws_link': torch.tensor([phrase_tags[ind_sent][ind]]), "dtype": torch.tensor([4])})
                    ####print("1", pid2nid[phrase_after_ind[ind_sent][ind][1]], pid2nid[phrase_id[1]])
                if phrase_id[0] == 0 and phrase_after_ind[ind_sent][ind][0] == 1:
                    ####print("22222222222222222222222222222222222222222222222222222222222222222")
                    #print(len(sent_before), len(phrase_after_ind[ind_sent]), ind, len(phrase_tags[ind_sent]), ind)
                    G.add_edges(pid2nid[phrase_id[1]], wid2nid[phrase_after_ind[ind_sent][ind][1]], data={'ws_link': torch.tensor([phrase_tags[ind_sent][ind]]), "dtype": torch.tensor([1])})   #类型是wid2nid[phrase_after_ind[ind_sent][ind][1]
                    if dict_pid.get(pid2nid[phrase_id[1]], 0) == 0:
                        dict_pid[pid2nid[phrase_id[1]]] = [phrase_after_ind[ind_sent][ind][1]]
                    #.append(pid2nid[phrase_id[1]])
                    else:
                        dict_pid[pid2nid[phrase_id[1]]].append(phrase_after_ind[ind_sent][ind][1])
                    #print("pid2nid1", pid2nid[phrase_id[1]], phrase_after_ind[ind_sent][ind][1])
                    G.add_edges(wid2nid[phrase_after_ind[ind_sent][ind][1]], pid2nid[phrase_id[1]], data={'ws_link': torch.tensor([phrase_tags[ind_sent][ind]]), "dtype": torch.tensor([4])})
                    ####print("2", wid2nid[phrase_after_ind[ind_sent][ind][1]], pid2nid[phrase_id[1]])
        #print(dict_pid)
              
        for ind_sent, sent_be in enumerate(sent_before_ind):
            for ind,  phrase_id in enumerate(sent_be):
                if phrase_id[0] == 0 and sent_after_ind[ind_sent][ind][0] == 0:
                    G.add_edges(sid2nid[phrase_id[1]], pid2nid[sent_after_ind[ind_sent][ind][1]], data={'ws_link': torch.tensor([phrase_tags[ind_sent][ind]]), "dtype": torch.tensor([3])})   
                    G.add_edges(pid2nid[sent_after_ind[ind_sent][ind][1]], sid2nid[phrase_id[1]], data={'ws_link': torch.tensor([phrase_tags[ind_sent][ind]]), "dtype": torch.tensor([13])})
                if phrase_id[0] == 0 and sent_after_ind[ind_sent][ind][0] == 1:
                    
                    G.add_edges(sid2nid[phrase_id[1]], wid2nid[sent_after_ind[ind_sent][ind][1]], data={'ws_link': torch.tensor([phrase_tags[ind_sent][ind]]), "dtype": torch.tensor([3])})   #类型是wid2nid[phrase_after_ind[ind_sent][ind][1]
                    G.add_edges(wid2nid[sent_after_ind[ind_sent][ind][1]], sid2nid[phrase_id[1]], data={'ws_link': torch.tensor([phrase_tags[ind_sent][ind]]), "dtype": torch.tensor([13])})
              
        p_node_id = [pid2nid[phrase_word_merge_ids[i]] for i in range(0, len(phrase_word))]
        p_node_id_id = [phrase_word_merge_ids[i] for i in range(0, len(phrase_word))]
         #= [pid2nid[phrase_before_only_ind] for i in range(0, len(phrase_before_only_ind[0]))]
        plo_node_id, pro_node_id, plo_node_id_id, pro_node_id_id = [], [], [], []

        for ind, sent_only in enumerate(phrase_before_only_ind[0]):
            #for ind_ph, phrase_ol in enumerate(sent_only):
            if sent_only[0] == 0:
                plo_node_id.append(pid2nid[sent_only[1]])
                plo_node_id_id.append(sent_only[1])
            elif sent_only[0] == 1:
                plo_node_id.append(wid2nid[sent_only[1]])
                plo_node_id_id.append(sent_only[1])
            else:
                #print("1", sid2nid[sent_only[1]])
                plo_node_id.append(sid2nid[sent_only[1]])
                plo_node_id_id.append(sent_only[1])
        for ind, sent_only in enumerate(phrase_before_only_ind[1]):
            #for ind_ph, phrase_ol in enumerate(sent_only):
            if sent_only[0] == 0:
                pro_node_id.append(pid2nid[sent_only[1]])
                pro_node_id_id.append(sent_only[1])
            elif sent_only[0] == 1:
                pro_node_id.append(wid2nid[sent_only[1]])
                pro_node_id_id.append(sent_only[1])
            else:
                #print("2", sid2nid[sent_only[1]])
                pro_node_id.append(sid2nid[sent_only[1]])
                pro_node_id_id.append(sent_only[1])

        x_node_id = [sid2nid[sentence_ids_list[0]]]
        x_node_id_id = [sentence_ids_list[0]]
        y_node_id = [sid2nid[sentence_ids_list[1]]]
        y_node_id_id = [sentence_ids_list[1]]
        label_phrase = []
        pl_node_id, pr_node_id = [], []
        if label == 'contradiction':
            if len(phrase_VP) == 2 and phrase_VP[0][0] == 0 and phrase_VP[1][0] == 1 and phrase_after_ind[phrase_VP[0][0]][phrase_VP[0][1]][0] == 0 and phrase_after_ind[phrase_VP[1][0]][phrase_VP[1][1]][0] == 0:
                pl_node_id.append(pid2nid[phrase_after_ind[phrase_VP[0][0]][phrase_VP[0][1]][1]])
                pr_node_id.append(pid2nid[phrase_after_ind[phrase_VP[1][0]][phrase_VP[1][1]][1]])
                label_phrase.append(label)
        if label == 'neutral':
            if len(phrase_VP) == 2 and phrase_VP[0][0] == 0 and phrase_VP[1][0] == 1 and phrase_after_ind[phrase_VP[0][0]][phrase_VP[0][1]][0] == 0 and phrase_after_ind[phrase_VP[1][0]][phrase_VP[1][1]][0] == 0:
                pl_node_id.append(pid2nid[phrase_after_ind[phrase_VP[0][0]][phrase_VP[0][1]][1]])
                pr_node_id.append(pid2nid[phrase_after_ind[phrase_VP[1][0]][phrase_VP[1][1]][1]])
                label_phrase.append(label)
        if label == 'entailment':
            if len(phrase_VP) == 2 and phrase_VP[0][0] == 0 and phrase_VP[1][0] == 1 and phrase_after_ind[phrase_VP[0][0]][phrase_VP[0][1]][0] == 0 and phrase_after_ind[phrase_VP[1][0]][phrase_VP[1][1]][0] == 0:
                #print("1", phrase_VP[0][1])
                #print("2", phrase_after_ind[phrase_VP[0][0]][phrase_VP[0][1]][1],pid2nid[phrase_after_ind[phrase_VP[0][0]][phrase_VP[0][1]][1]])
                pl_node_id.append(pid2nid[phrase_after_ind[phrase_VP[0][0]][phrase_VP[0][1]][1]])

                pr_node_id.append(pid2nid[phrase_after_ind[phrase_VP[1][0]][phrase_VP[1][1]][1]])
                label_phrase.append(label)

        
        # word - NER type
        snode_id = G.filter_nodes(lambda nodes: nodes.data['dtype'] == 2) 
        #print(snode_id.shape)
        #G.nodes[snode_id]
        G.nodes[snode_id].data['init_state_s'] = torch.zeros([2,300])
        
        return x_node_id, y_node_id, pl_node_id, pr_node_id, p_node_id, plo_node_id, pro_node_id, x_node_id_id, y_node_id_id, p_node_id_id, plo_node_id_id, pro_node_id_id, phrase_word_merge, label_phrase, G

    def __getitem__(self, index):
        """
        .. note:: `utter` and `u` both stands for utterance
        """
        item = {}
        

        item["utters"] = self.conv_to_ids(self.data[index]['feats']['sent_nophrase'])
        item["u_lengths"] = [len(o) for o in item["utters"]]    #计算每个句子的长度，方便mask

        item["u_masks"] = [torch.LongTensor([1 for _ in range(o)]) for o in item["u_lengths"]] #对句子进行mask,弄成同一长度以方便LSTM对句子进行编码对，每个句子进行

        item['rids'] = self.label_to_oneHot(self.data[index]['label'])   #标签  
              #把label表示成onehot的形式
        item['rids_p'] = self.label_to_oneHot(self.data[index]['label']) 
        
        #print(item['rids'])
        #print(self.data[index]['feats']['tokens_pos'])
        #for o in self.data[index]['feats']['tokens_pos']:
            #for oo in o:
                #print(oo)
                #print(constant.pos_s2i[oo])
        item['pos_tag'] = [torch.LongTensor([constant.pos_s2i[oo] for oo in o])  for o in self.data[index]['feats']['tokens_pos'] ]       #词性标签转换成标签  #前面是把词，词性
        item['phrase_tag'] = [torch.LongTensor([constant.pos_s2i[oo] for oo in o])  for o in self.data[index]['feats']['phrase_tag'] ]       #词性标签转换成标签   #前面是把词，词性，
        item['phrase_pos'] = [torch.LongTensor([constant.pos_s2i[oo] for oo in o])  for o in self.data[index]['feats']['phrase_pos'] ]  

        item['sent_pos'] = [torch.LongTensor([constant.pos_s2i[oo] for oo in o])  for o in self.data[index]['feats']['sent_pos'] ]  
        item['x_node_id'], item['y_node_id'], item['pl_node_id'], item['pr_node_id'],item['p_node_id'], item['plo_node_id'], item['pro_node_id'], item['x_node_id_id'], item['y_node_id_id'], item['p_node_id_id'], item['plo_node_id_id'], item['pro_node_id_id'], phrase_word_merge, label_phrase, item['conv_graph'] = self.creat_graph(
            self.data[index]['feats']['tokens_sentence'],
            self.data[index]['feats']['phrase_before'],
            self.data[index]['feats']['phrase_after'],
            self.data[index]['feats']['phrase_tag'],
            self.data[index]['feats']['phrase'],
            self.data[index]['feats']['phrase_word'],
            self.data[index]['feats']['tokens_phrase'],
            self.data[index]['feats']['words'],
            self.data[index]['feats']['tags'],
            self.data[index]['feats']['tokens_word'],
            self.data[index]['feats']['tokens_pos'],
            self.data[index]['feats']['words2tags'],
            self.data[index]['feats']['tags2words'],
            self.data[index]['feats']['sent_id'],
            self.data[index]['feats']['word_el'],  #'sent_nophrase':[], 'sent_before':[], 'phrasenosent_after':[], 'sent_after':[], 'phrasenosent_before':[]
            self.data[index]['feats']['sent_nophrase'],
            self.data[index]['feats']['sent_before'],
            self.data[index]['feats']['phrasenosent_after'],
            self.data[index]['feats']['sent_after'],
            self.data[index]['feats']['phrasenosent_before'],
            self.data[index]['label'],
            self.data[index]['feats']['phrase_pos'],
            self.data[index]['feats']['phrase_tag'],
            [ [constant.pos_s2i[oo] for oo in o] for o in self.data[index]['feats']['sent_pos'] ],
            [ [constant.pos_s2i[oo] for oo in o] for o in self.data[index]['feats']['tags'] ],
            [ [constant.pos_s2i[oo] for oo in o] for o in self.data[index]['feats']['phrase_tag'] ],
            [ [constant.pos_s2i[oo] for oo in o] for o in self.data[index]['feats']['phrase_pos'] ],
        ) 

        if item['pl_node_id'] == []:
            item["phrase_ind"] = 0
        if item['pl_node_id'] != []:
            item["phrase_ind"] = 1
        #print(item["index"])
        item["phrases"] = self.conv_to_ids(phrase_word_merge)     #对每个单词进行编号
        item["p_lengths"] = [len(o) for o in item["phrases"]]    #计算每个短语的长度，方便mask
        item["p_masks"] = [torch.LongTensor([1 for _ in range(o)]) for o in item["p_lengths"]] 
        return item

    def get_arg_pos(self, arg, conv):
        arg_positions = []
        arg = arg.lower().split()
        for utter in conv:
            for i, w in enumerate(utter):
                if w == arg[0]:
                    arg_positions.append( torch.LongTensor(get_positions(i, i+len(arg)-1, len(utter))) )
                    break
                elif i == len(utter) - 1:
                    arg_positions.append( torch.LongTensor(get_positions(1, 0, len(utter))) )

        return arg_positions



def collate_fn(data):
    """
    .. note:: `utter` for utterance, `conv` for conversation, `seq` for sequence
    """
    items = {}
    #print(data[0])
    for k in data[0].keys():
        items[k] = [d[k] for d in data]
    
    num_utters = [len(conv) for conv in items['utters']] 
    num_phrase = [len(conv) for conv in items['phrases']] 
    max_utters = max(num_utters)
    max_phrases = max(num_phrase)
    max_seq = max([utter_l for conv_l in items['u_lengths'] for utter_l in conv_l])
    max_p_seq = max([utter_l for conv_l in items['p_lengths'] for utter_l in conv_l])

    def pad(convs, conv_lengths=items['u_lengths'], max_utters=max_utters, max_seq=max_seq):
        """
        Parameters
        ----------
        convs: list[list[list[int]]]
        conv_lengths: list[list[int]] 
        max_utters: int
            the max number of utterance in one conversation
        max_seq: int
            the max number of words in one utterance
        
        Returns
        -------
        padded_convs: (batch size, max number of utterance, max sequence length)
        .. note:: pad index is 0
        """
        padded_convs = torch.zeros(len(convs), max_utters, max_seq, dtype=torch.int64)
        for b_i, (conv_l, conv) in enumerate(zip(conv_lengths, convs)):
            for u_i, (utter_l, utter) in enumerate(zip(conv_l, conv)):
                #print(conv_l, conv)
                #print(b_i, u_i, utter_l, len(utter))
                padded_convs[b_i, u_i, :utter_l] = utter
        return padded_convs #.to(config.device)
    def lenth_t(conv_lenth_s):
        conv_lenth_s_list, conv_lenth_p_list =[], []
        for sent in conv_lenth_s:
            lenths = []
            for lenth in sent:
                lenths.append([1/lenth]*300)
            conv_lenth_s_list.append(lenths)
        return conv_lenth_s_list
    def sub2fullgraph(node_id, werid, dict_id_reverse):
        node_id_full = []
        for sent_id in node_id:
            node_id_full.append([])
            for id in sent_id:
                for ind_wid, id_w in enumerate(werid): 
                    if id == id_w:
                        if dict_id_reverse.get(ind_wid, 0) == 0:
                            node_id_full[-1].append(ind_wid)
                        else:
                            node_id_full[-1].append(dict_id_reverse[ind_wid])
                        break
        return node_id_full
    def graph_remove(items_graph, x_node_id_id, y_node_id_id, p_node_id_id, plo_node_id_id, pro_node_id_id):
        batch_graph1 = dgl.batch(items_graph)
        #print(items['conv_graph'][0])
        #wnode_id = batch_graph.filter_nodes(lambda nodes: nodes.data['dtype'] == 0)
        wnode_id = batch_graph1.nodes()
        werid = batch_graph1.nodes[wnode_id].data['id'] 
        #edges()
  
                
        werid = werid.tolist()

                        
        #print("111", len(werid))
        dict_id, dict_id_reverse = {}, {}
        addition_src, addition_dst, edges_id, repetition, ws_link_re, edge_dtype_re, temp_add_src, temp_add_dst, ws_link_re_temp, edge_dtype_re_temp, del_temp  = [], [], [], [], [], [], [], [], [], [], []
        for ind_wid, id in enumerate(werid):
            #print(len(werid))
            if id not in repetition: # and dict_id.get(ind_wid, 0) == 0:
                repetition.append(id)
                repetition_id = []
                #dict_id[ind_wid] =[]
                #print(ind_wid+1, len(werid))
                for ind_wid_1, id_1 in enumerate(werid[ind_wid+1:len(werid)]):
                    if id == id_1:
                        dict_id_reverse[ind_wid_1 + ind_wid + 1] = ind_wid
                        repetition_id.append(ind_wid_1 + ind_wid + 1)
                        repetition.append(id_1)
                        
                        #dict_id[ind_wid].append(ind_wid_1)
                if len(repetition_id) != 0:
                    dict_id[ind_wid] = repetition_id
        x_node_id_full = sub2fullgraph(x_node_id_id, werid, dict_id_reverse)
        y_node_id_full = sub2fullgraph(y_node_id_id, werid, dict_id_reverse)
        p_node_id_full = sub2fullgraph(p_node_id_id, werid, dict_id_reverse)
        plo_node_id_full = sub2fullgraph(plo_node_id_id, werid, dict_id_reverse)
        pro_node_id_full = sub2fullgraph(pro_node_id_id, werid, dict_id_reverse)
        d['dict_id'] = dict_id
        #print(len(dict_id), dict_id)
        l = batch_graph1.edges()
        ws_link = batch_graph1.edata["ws_link"]
        edge_dtype = batch_graph1.edata["dtype"]
        ws_link = ws_link.tolist()
        edge_dtype = edge_dtype.tolist()
        edges_l = l[0].tolist()
        edges_r = l[1].tolist()
        #print("123321", len(edges_l))
        #torch.LongTensor


        #print(len(edges_l), len(graph_edges_l))
        wnode_id_list = wnode_id.tolist()
        wnode_id_list = torch.LongTensor(wnode_id_list)
        #print("1111", dict_id)
        
        for ind_key, key in enumerate(dict_id.keys()):
            #print("22222", len(dict_id.keys()))
            #print("333333333", len(edges_l))
            for id in dict_id[key]:
                for ind_edgl, edge_l in enumerate(edges_l):
                    
                    if id == edge_l:
                        addition_src.append(key)
                        addition_dst.append(edges_r[ind_edgl])
                        edges_id.append(ind_edgl)
                        ws_link_re.append(ws_link[ind_edgl])
                        edge_dtype_re.append(edge_dtype[ind_edgl])
                for ind_edgr, edge_r in enumerate(edges_r):
                    if id == edge_r:
                        addition_src.append(edges_l[ind_edgr])
                        addition_dst.append(key)
                        edges_id.append(ind_edgr)
                        ws_link_re.append(ws_link[ind_edgr])
                        edge_dtype_re.append(edge_dtype[ind_edgr])
                for ind_edg_re, edge_l in enumerate(addition_src):
                    if id == edge_l:
                        temp_add_src.append(key)
                        temp_add_dst.append(addition_dst[ind_edg_re])
                        ws_link_re_temp.append(ws_link_re[ind_edg_re])
                        edge_dtype_re_temp.append(edge_dtype_re[ind_edg_re])
                        del_temp.append(ind_edg_re)
                        # del addition_src[ind_edg_re]
                        # del addition_dst[ind_edg_re]
                        # del ws_link_re[ind_edg_re]
                        # del edge_dtype_re[ind_edg_re]
                for ind_edgr_re, edge_r in enumerate(addition_dst):
                    if id == edge_r:
                        temp_add_src.append(addition_src[ind_edgr_re])
                        temp_add_dst.append(key)
                        ws_link_re_temp.append(ws_link_re[ind_edgr_re])
                        edge_dtype_re_temp.append(edge_dtype_re[ind_edgr_re])
                        del_temp.append(ind_edgr_re)
                        # del addition_src[ind_edgr_re]
                        # del addition_dst[ind_edgr_re]
                        # del ws_link_re[ind_edgr_re]
                        # del edge_dtype_re[ind_edgr_re]
                del_temp = sorted(del_temp, reverse=True)

                addition_src = addition_src + temp_add_src
                addition_dst = addition_dst + temp_add_dst
                ws_link_re = ws_link_re + ws_link_re_temp
                edge_dtype_re = edge_dtype_re + edge_dtype_re_temp
                for temp in del_temp:
                    #print(temp)
                    del addition_src[temp]
                    del addition_dst[temp]
                    del ws_link_re[temp]
                    del edge_dtype_re[temp]
        
                #print(temp_add_src)
                del_temp = []
                temp_add_src = []
                temp_add_dst = []
                ws_link_re_temp = []
                edge_dtype_re_temp = []
                #for 
        #print(len(addition_src))
        #print(len(edges_id))
        #print("6665555", len(dict_id.keys()))
        #print(len(edges_id))
        batch_graph1.remove_edges(edges_id)
        for ind_src, src in enumerate(addition_src):
            batch_graph1.add_edges(src, addition_dst[ind_src], data={'ws_link': torch.tensor([ws_link_re[ind_src]]), "dtype": torch.tensor([edge_dtype_re[ind_src]])})
        graph_edges = batch_graph1.edges()
        graph_edges_l = graph_edges[0].tolist()
        #print("222", len(graph_edges_l))
        #print(p_node_id_full)
        #edge_dtype = batch_graph1.edata["dtype"]
        #print()
        #s1edge_id = g.filter_edges(lambda edges: edges.data["dtype"] == 4)
        #l = batch_graph1.edges()
        #for ind_ed, type in enumerate(s1edge_id):
            #if type == 4:
            #print(l[0][type], l[1][type])
            
        return batch_graph1, x_node_id_full, y_node_id_full, p_node_id_full, plo_node_id_full, pro_node_id_full
    def pad_p(convs, conv_lengths=items['p_lengths'], max_phrases=max_phrases, max_p_seq=max_p_seq):
        """
        Parameters
        ----------
        convs: list[list[list[int]]]
        conv_lengths: list[list[int]] 
        max_utters: int
            the max number of utterance in one conversation
        max_seq: int
            the max number of words in one utterance
        
        Returns
        -------
        padded_convs: (batch size, max number of utterance, max sequence length)
        .. note:: pad index is 0
        """
        padded_convs = torch.zeros(len(convs), max_phrases, max_p_seq, dtype=torch.int64)
        for b_i, (conv_l, conv) in enumerate(zip(conv_lengths, convs)):
            for u_i, (utter_l, utter) in enumerate(zip(conv_l, conv)):
                padded_convs[b_i, u_i, :utter_l] = utter
        return padded_convs #.to(config.device)
    def pad_length(conv_lengths):
        """
        conv_lengths: list[list[int]] 
        """
        padded_lengths = torch.zeros((len(conv_lengths), max_utters), dtype=torch.int64)
        for bi, conv_l in enumerate(conv_lengths):
            for ui, u_l in enumerate(conv_l):
                padded_lengths[bi, ui] = u_l
        return padded_lengths
    def pad_p_length(conv_lengths):
        """
        conv_lengths: list[list[int]] 
        """
        padded_lengths = torch.zeros((len(conv_lengths), max_phrases), dtype=torch.int64)
        for bi, conv_l in enumerate(conv_lengths):
            for ui, u_l in enumerate(conv_l):
                padded_lengths[bi, ui] = u_l
        return padded_lengths

    d = {}
    d['utter_index'] = [oo + i*max_utters for i, o in enumerate(num_utters) for oo in range(o)]
    d['utter_p_index'] = [oo + i*max_phrases for i, o in enumerate(num_phrase) for oo in range(o)]
    d['conv_batch'], d['conv_mask'] = pad(items['utters']), pad(items['u_masks'])
    d['conv_p_batch'], d['conv_p_mask'] = pad_p(items['phrases']), pad_p(items['p_masks'])

    d['phrase_pos'] = pad_p(items['phrase_pos'])#, pad(items['pos_tag']) #d['ner_type'],  pad(items['ner_type']), 
    d['sent_pos'] = pad(items['sent_pos'])

    d['rids'] = torch.stack(items['rids'])
    d['rids_p'] = torch.stack(items['rids_p'])     #
    d['phrase_ind'] = items['phrase_ind']
    #d['batch_graph'] = dgl.batch(items['conv_graph'])
    d['batch_graph1'], d['x_node_id_full'], d['y_node_id_full'], d['p_node_id_full'], d['plo_node_id_full'], d['pro_node_id_full'] = graph_remove(items['conv_graph'], items['x_node_id_id'], items['y_node_id_id'], items['p_node_id_id'], items['plo_node_id_id'], items['pro_node_id_id'])

    d['x_node_id'], d['y_node_id'], d['p_node_id'], d['pl_node_id'], d['pr_node_id'], d['plo_node_id'], d['pro_node_id'],d['x_node_id_id'], d['y_node_id_id'], d['p_node_id_id'], d['plo_node_id_id'], d['pro_node_id_id'] = items['x_node_id'], items['y_node_id'], items['p_node_id'], items['pl_node_id'], items['pr_node_id'], items['plo_node_id'], items['pro_node_id'], items['x_node_id_id'], items['y_node_id_id'], items['p_node_id_id'], items['plo_node_id_id'], items['pro_node_id_id']

    return d


