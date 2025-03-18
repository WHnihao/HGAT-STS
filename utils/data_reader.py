import os
import random
import numpy as np
import torch.nn as nn
from sklearn.decomposition import PCA
import pickle
import json
import argparse
import time
import numpy as np
from collections import Counter
from model.bert import Bertsent
import spacy

from utils import constant
from utils.config import config

#nlp = spacy.load('en_core_web_sm')

def build_embedding(wv_file, vocab, wv_dim):
    vocab_size = len(vocab)
    emb = np.random.randn(vocab_size, config.embed_dim) * 0.01
    emb[constant.PAD_ID] = 0 # <pad> should be all 0

    w2id = {w: i for i, w in enumerate(vocab)}       #词或者短语对应唯一的id
    with open(wv_file, encoding="utf8") as f:
        for line in f:
            elems = line.split()
            token = ''.join(elems[0:-wv_dim])     #这个是取单词？
            if token in w2id:
                emb[w2id[token]] = [float(v) for v in elems[-wv_dim:]]  #这个是单词和句子对应
    return emb





class Vocab(object):
    def __init__(self, init_wordlist, word_counter):
        self.word2id = {w:i for i,w in enumerate(init_wordlist)}    #初始的单词单词和id的对应
        self.id2word = {i:w for i,w in enumerate(init_wordlist)}    #id和单词的对应
        self.n_words = len(init_wordlist)                #初始单词词典的长度
        self.word_counter = word_counter                 #单词计数

    def map(self, token_list):
        """
        Map a list of tokens to their ids.              #
        """
        return [self.word2id[w] if w in self.word2id else constant.UNK_ID for w in token_list]    #UNK和单词的分别统计

    def unmap(self, idx_list):
        """
        Unmap ids back to tokens.
        """
        return [self.id2word[idx] for idx in idx_list]
    

class Vocab_phrase(object):
    def __init__(self, init_phraselist, init_wordlist):
        self.phrase2id = {' '.join(word for word in w):i + 100000 for i,w in enumerate(init_phraselist)}    #初始的单词单词和id的对应len(init_wordlist)
        self.id2phrase = {i + 100000:' '.join(word for word in w) for i,w in enumerate(init_phraselist)}    #id和单词的对应len(init_wordlist)
        self.n_words = len(init_phraselist)                #初始单词词典的长度                 #单词计数

    def map(self, token_list):
        """
        Map a list of tokens to their ids.              #
        """
        return [self.phrase2id[' '.join(word for word in w)] for w in token_list]    #UNK和单词的分别统计

    def unmap(self, idx_list):
        """
        Unmap ids back to tokens.
        """
        return [self.id2phrase[idx] for idx in idx_list]

        
class Vocab_sent(object):
    def __init__(self, init_sentlist, init_phraselist, init_wordlist):
        #' '.join(word[0] for word in word_list)
        self.sent2id = {' '.join(word for word in w):i + len(init_phraselist) + 100000 for i,w in enumerate(init_sentlist)}    #初始的单词单词和id的对应len(init_wordlist)
        self.id2sent = {i + len(init_phraselist) + 100000:' '.join(word for word in w) for i,w in enumerate(init_sentlist)}    #id和单词的对应
        self.n_sent = len(init_sentlist)                #初始单词词典的长度                 #单词计数

    def map(self, token_list):
        """
        Map a list of tokens to their ids.              #
        """
        return [self.sent2id[' '.join(word for word in w)] for w in token_list]    #UNK和单词的分别统计

    def unmap(self, idx_list):
        """
        Unmap ids back to tokens.
        """
        return [self.id2sent[idx] for idx in idx_list]

def get_feats(utters, word_pairs):
    #ret = {'tokens': [], 'dep_head': [], 'dep_tag':[], 'pos_tag':[], 'ner_iob':[], 'ner_type':[], 'noun_chunks':[], 'noun_chunks_root':[], 'tokens_sentence':[]}
    ret = {'tokens_sentence': [], 'phrase_before': [], 'phrase_after': [], 'phrase_tag': [], 'phrase': [], 'phrase_pos': [], 'phrase_word': [], 'tokens_phrase': [], 'words': [], 'tags': [], 'tokens_word': [], 'tokens_pos': [[], []], 'words2tags': [[],[]], 'tags2words': [[],[]], 'sent_id':[], 'word_el':[], 'phrase_sent':[[],[]], 'sent_nophrase':[], 'sent_before':[], 'phrasenosent_after':[], 'sent_after':[], 'phrasenosent_before':[], 'sent_pos':[]}
    #'words'， 'tags' 'phrase'
    

    for phrase in word_pairs[0]:
        if phrase[0][0].find(" ") ==-1 and phrase[0][0] not in ret['words2tags'][0]:
            ret['words2tags'][0].append(phrase[0][1])
            ret['tags2words'][0].append(phrase[1])
    #print(ret['words2tags'][0])
    #for phrase in word_pairs[0]:
        if phrase[0][1].find(" ") ==-1 and phrase[0][1] not in ret['words2tags'][0]:
            ret['words2tags'][0].append(phrase[0][1])
            ret['tags2words'][0].append(phrase[1])
    for phrase in word_pairs[1]:
        #print()
        if phrase[0][0].find(" ") ==-1 and phrase[0][0] not in ret['words2tags'][1]:
            ret['words2tags'][1].append(phrase[0][1])
            ret['tags2words'][1].append(phrase[1])
            #print(phrase)
    #for phrase in word_pairs[1]:
        #print(phrase[0][1])
        #print(ret['words2tags'][1])
        if phrase[0][1].find(" ") ==-1 and phrase[0][1] not in ret['words2tags'][1]:
            ret['words2tags'][1].append(phrase[0][1])
            ret['tags2words'][1].append(phrase[1])
    for ind_u, utter in enumerate(utters[0]):
        # ret['tags'].append([])
        # utter = nlp(utter)                #对句子进行句法分析
        utter = utter.split(" ")

        ret['tokens_sentence'].append([token for token in utter])

    for ind, sent in enumerate(ret['tokens_sentence']):
        for word in sent:
            if word not in ret['word_el']:
                ret['word_el'].append(word)
    sent_ind = [[],[]]
    for ind, sent in enumerate(word_pairs):
        a = []
        ret['sent_before'].append([])
        ret['phrasenosent_before'].append([])
        ret['phrase_before'].append([phrase[0][0] for phrase in sent])     #phrase虽然在第一个，但是仍然有单个词的情况，phrase_before表示，
        for phrase in ret['phrase_before'][-1]:
            phrase = phrase.split(" ")
            a.append(phrase)
        ret['phrase_before'][-1] = a
        for ind_ph, phrase in enumerate(ret['phrase_before'][-1]):
            if phrase == ret['tokens_sentence'][0] or phrase == ret['tokens_sentence'][1]:
                ret['sent_before'][-1].append(phrase)
                sent_ind[ind].append(ind_ph)
            else:
                ret['phrasenosent_before'][-1].append(phrase)
    #print("sent_before", ret['sent_before'])
    ####for ind, sent in enumerate(word_pairs):
        a = []
        ret['sent_after'].append([])
        ret['phrasenosent_after'].append([])
        ret['phrase_after'].append([phrase[0][1] for phrase in sent])     
        for phrase in ret['phrase_after'][-1]:
            phrase = phrase.split(" ")
            a.append(phrase)
        ret['phrase_after'][-1] = a
        for ind_ph, phrase in enumerate(ret['phrase_after'][-1]):
            if ind_ph in sent_ind[ind]: # or phrase == ret['words'][0]:
                ret['sent_after'][-1].append(phrase)
                #sent_ind[ind].apend(ind_ph)
            else:
                ret['phrasenosent_after'][-1].append(phrase)
    #print('phrasenosent_after', ret['phrasenosent_after'])
    ####for sent in word_pairs:
        ret['phrase_tag'].append([phrase[1] for phrase in sent])     #phrase虽然在第一个，但是仍然有单个词的情况
    #print(ret['phrase_tag'])
    # for sent in word_pairs:
        # for phrase in sent:
            # if phrase[0][0].find(" ") != -1 and phrase[0][0] not in ret['phrase']:
                # ret['phrase'].append(phrase[0][0])
        #ret['phrase'].append([phrase[0][0] for phrase in sent] )
    #print("ret['phrase']", ret['phrase'])
    for sent in ret['phrasenosent_before']:
        for phrase in sent:
            if phrase not in ret['phrase_word'] and len(phrase) >1:
                ret['phrase_word'].append(phrase)
    #print("phrase_word", ret['phrase_word'])
    for sent in ret['sent_before']:
        for phrase in sent:
            if phrase not in ret['sent_nophrase'] and len(phrase) >1:
                ret['sent_nophrase'].append(phrase)
    #print("sent_nophrase", ret['sent_nophrase'])
    for ind_u, utter in enumerate(ret['phrase_word']):
        ret['phrase_pos'].append([])
        #utter = nlp(utter)                #对句子进行句法分析
        #utter = utter.split(" ")
        #print("123456", utter)
        #print(utter)
        for word in utter:
            a = 0
            #print(ret['words2tags'][0])
            for ind, wordtag in enumerate(ret['words2tags'][0]):
                if word == wordtag:
                    ret['phrase_pos'][-1].append(ret['tags2words'][0][ind])
                    a = 1 
                    #break
            if a == 0:
                for ind, wordtag in enumerate(ret['words2tags'][1]):
                    if word == wordtag:
                        ret['phrase_pos'][-1].append(ret['tags2words'][1][ind]) 
    for ind_u, utter in enumerate(ret['sent_nophrase']):
        ret['sent_pos'].append([])
        for word in utter:
            a = 0
            #print(ret['words2tags'][0])
            for ind, wordtag in enumerate(ret['words2tags'][0]):
                if word == wordtag:
                    ret['sent_pos'][-1].append(ret['tags2words'][0][ind])
                    a = 1 
                    #break
            if a == 0:
                for ind, wordtag in enumerate(ret['words2tags'][1]):
                    if word == wordtag:
                        ret['sent_pos'][-1].append(ret['tags2words'][1][ind]) 
                        #break
    #for ind, phrase in enumerate(ret['phrase_word']):
        #if len(phrase) !=len(ret['phrase_pos'][ind]):
            #print("ret['phrase_word']", len(phrase), len(ret['phrase_pos'][ind]))
            #print("ret['phrase_word']", phrase, ret['phrase_pos'][ind])
        #print("ret['phrase_pos']", len(ret['phrase_pos'][ind]))
    #print("ret['phrase_word']", ret['phrase_word'])
    #print("ret['phrase_pos']", ret['phrase_pos'])
    #print(ret['phrase_word'])
    #print("5678", ret['phrase_before'])
    #print("1234", ret['phrase_after'])
    #for ind, phrase in enumerate(ret['phrase_before'][0]):
        #if phrase == ret['words'][0]:
            #ret['sent_id'][0].append(ind)
            #print(phrase)
            #print(ret['words'])
    for ind, phrase in enumerate(ret['phrase_word']):
        if phrase == ret['tokens_sentence'][0] or phrase == ret['tokens_sentence'][1]:
            ret['sent_id'].append(ind)
    
    #if len(ret['sent_id']) < 2:
        #print(ret['sent_id'])
        #print("ret['words']", ret['words'])
        #print("utters", utters) 
        #print("word_pairs", word_pairs)
    #print(ret['sent_id'])
    #for phrase in ret['phrase'][0]:
        #print("123", phrase)
        #utter = nlp(phrase)
        #utter = phrase.split(" ")
        #ret['tokens_phrase'].append([str(token) for token in utter]) 
        #ret['tokens_phrase'].append([str(token) for token in utter])
        #print(ret['tokens_phrase'])
    #ret['words'].append([word[0][1] for word in word_pairs])
    #ret['tags'].append([tags[1][0] for tags in word_pairs])
    #这三个是短语和短语，
    #print("word", ret['words'])
    #print("tokens_sentence", ret['tokens_sentence'])

    for phrase in word_pairs[0]:   #这后面短语和单词，短语和短语是分开的。分开表示                    #word_pairs中有

        if phrase[0][1].find(" ") ==-1:                   #只可能1作为单词，一般phrase[0][0]为短语或句子
            ret['tokens_phrase'].append([phrase[0][0]])
            ret['tokens_word'].append([phrase[0][1]])
            #ret['tokens'].append([phrase[0][1]])    #
            ret['tokens_pos'][0].append(phrase[1])   #应该算是词性标注，存储词性标注
    for phrase in word_pairs[1]:   #这后面短语和单词，短语和短语是分开的。分开表示                    #word_pairs中有

        if phrase[0][1].find(" ") ==-1:                   #只可能1作为单词，一般phrase[0][0]为短语或句子
            ret['tokens_phrase'].append([phrase[0][0]])
            ret['tokens_word'].append([phrase[0][1]])
            #ret['tokens'].append([phrase[0][1]])    #
            ret['tokens_pos'][1].append(phrase[1])   #应该算是词性标注，存储词性标注
        # if phrase[0][0] ==utters[0][1] or phrase[0][0] ==utters[0][1]:
            # ret['tokens_pos'].append([phrase[1]])
            
            # ret['sentence'].append([phrase[0][0]]) 
        #if phrase[0][1].find(" ") !=-1:               #区分是短语还是单词，!=-1是说明是短语，以下应该是短语之间的关系
            #for word in phrase[0][1]:
                #word_ind = ret['tokens'].index(word)
            #ret['phrase_before'].append([phrase[0][0]])                       #存储短语
            #ret['phrase_after'].append([phrase[0][1]]) 
            ####ret['phrase_tag'].append([phrase[1]])
    #print("tokens_word", ret['tokens_word'])
    #print("tokens_word", ret['tokens_word'])
                #ret['phrase_pos'].append([ret['tokens_pos'][word_ind]])    #存储短语和词之间的标注     #单词和短语之间的链接，词性作为边的值
            #ret['phrase_tag'].append([phrase[1]])      #短语的之间的边的值是短语的标注
            #ret['phrase'].append([phrase[0][0]])
        #ret['tokens'].append([str(token) if token for token in utter])   #判断单词还是短语，把单词加入
        #ret['pos_tag'].append([str(token) for token in utter])            #自己加的，单词的词性标注
        #ret['phrase'].append([str(token) if token for token in utter])   #自己加的，短语
        #ret['phrase_tag'].append([str(token) for token in utter])            #自己加的，短语标记
        # ret['dep_head'].append( [token.head.i+1 if token.i != token.head.i else 0 for token in utter ])  #这个没有关注
        # ret['dep_tag'].append([token.dep_ for token in utter])
        # ret['pos_tag'].append( [token.pos_ for token in utter]
        # ret['ner_iob'].append([utter[i].ent_iob_ for i in range(len(utter))])  #这个是识别实体的开始和结束
        # ret['ner_type'].append([utter[i].ent_type_ if i!=0 else 'PERSON' for i in range(len(utter))]) # hard-code ner type to be 'PER' for speaker
        # ret['noun_chunks'].append([str(o) for o in utter.noun_chunks])
        # ret['noun_chunks_root'].append([str(o.root) for o in utter.noun_chunks])

    return ret                #对对话句子做一些分解
    

word_pairs = {"it's":"it is", "don't":"do not", "doesn't":"does not", "didn't":"did not", "you'd":"you would", "you're":"you are", "you'll":"you will", "i'm":"i am", "they're":"they are", "that's":"that is", "what's":"what is", "couldn't":"could not", "i've":"i have", "we've":"we have", "can't":"cannot", "i'd":"i would", "i'd":"i would", "aren't":"are not", "isn't":"is not", "wasn't":"was not", "weren't":"were not", "won't":"will not", "there's":"there is", "there're":"there are"}


#def load_data(filename):   #读入数据
   #tokens = []
    #word_pairs = {}
    #with open(filename) as infile:
        #data = json.load(infile)
    #D = []
    #for i in range(len(data)):
        #utters = data[i][0]   #有时间改一下
        #spacy_feats = get_feats(utters, word_pairs)
        #for j in range(len(data[i][1])):
           # d = {}
            #d['us1'] = sentence1
            #d['us2'] = sentence2
            #d['label'] = lable
            #d['us'] = utters
            #d['feats'] = spacy_feats                 
            # d['x_type'] = data[i][1][j]["x_type"]
            # d['y_type'] = data[i][1][j]["y_type"]
            # d['rid'] = data[i][1][j]["rid"]
            # d['r'] = data[i][1][j]["r"]
            # d['t'] = data[i][1][j]["t"]

            # d['x'] = [str(token) for token in nlp(data[i][1][j]["x"])]
            # d['x'] = ''.join(d['x']) if 'Speaker' in d['x'] else ' '.join(d['x'])
            # d['y'] = [str(token) for token in nlp(data[i][1][j]["y"])]
            # d['y'] = ''.join(d['y']) if 'Speaker' in d['y'] else ' '.join(d['y'])
            #D.append(d)
        
        #tokens += [oo for o in d['feats']['tokens'] for oo in o]

    #return tokens, D
def load_data(filename): 
    tokens = []  
    phrases = []
    sentence_list = []
    class_list = []
    word_pairs = {}
    start_time = time.time() 
    print("load data start time",  start_time)
    file = open(filename, "r", encoding = "utf-8")   #路径改一下
    lines = file.readlines()
    if filename == config.train1_f or filename == config.train2_f or filename == config.train3_f:
        #del lines[0]
        random.shuffle(lines)
    wordcount = {}
    phrasecount = {}
    sentencecount = {}
    tree_save1, tree_save2 = [], []
    local_save1, local_save2 = [], []
    
    for ind, line in enumerate(lines):
        ###print(ind)
        s = line.strip().split("	")
        classes = s[0]
        if ind == 0:
            print(s)
        sent1 = s[3]
        sent2 = s[4]
        sent1_pos, sent2_pos = [], []
        class_list.append(classes)
        #print(sent1)
        #if sentencecount.get(sent1, 0) == 0:
        sentencecount[sent1] = sentencecount.get(sent1, 0) + 1
        sentencecount[sent2] = sentencecount.get(sent2, 0) + 1
        sent1 = sent1.replace(")", " )")
        sent2 = sent2.replace(")", " )")
        #sent1 = sent1.replace("))", ") )")
        #
        sent1 = sent1.split(" ")
        sent2 = sent2.split(" ")
        prior_ind = 0
        ind_wipe = []
        sent1_pos = sent1_pos + sent1
        sent2_pos = sent2_pos + sent2
        #sent1 = sent1.lower()

        for ind, word in enumerate(sent1):
            if word.find("(") != -1:     #说明该字符串中包含"("
                ind_left = ind
                #if ind < len(sent1)-1 and sent1[ind+1].find("(") != -1:     #表示下一个字符串中也包含"("
                sent1[ind] = "("#sent1[ind][0]      #把包含"("的字符串转化为"("
                    #print(sent1[ind])     #这儿把所有标注都去掉不适合当前语料
                #else:
                    #ind_wipe.append(ind)  #否则把当前节点除去，说明当前节点是单词的词性标注加"("
            #if word.find(")") != -1 and word != ")":
                #sent1[ind] = sent1[ind].replace(")", "")      #排除干扰项
                
        #print(sent1)
        #for ind, i in enumerate(ind_wipe):
            #del sent1[ind_wipe[len(ind_wipe)-1 - ind]]
            #prior_ind = ind_left
        #print(sent1)
        for ind, word in enumerate(sent2):
            if word.find("(") != -1:     #说明该字符串中包含"("
                ind_left = ind
                #if ind < len(sent1)-1 and sent1[ind+1].find("(") != -1:     #表示下一个字符串中也包含"("
                sent2[ind] = "("
        root_node, root_node2 = [], []
        root_node_pos, root_node_pos2 = [], []
        root_node.append(sent1[1:len(sent1)-1])
        root_node2.append(sent2[1:len(sent2)-1])
        #print("0", root_node)
        root_node_pos.append(sent1_pos[1:len(sent1_pos)-1])
        root_node_pos2.append(sent2_pos[1:len(sent2_pos)-1])
        root_father = [[[0,0], 0, "", 1]]
        root_pos = []
        #print(root_node)
        #print(root_node)
        leaf_num = -1
        tree_sentence = []
        #tree_sentence
        tree_coord = []
        #print(root_node)
        tree_coord.append(root_father)
        tree_sentence.append(root_node)
        lay = 0
        number = 0
        
        #print(root_node)
        if root_node!=[[]]:
            while(leaf_num != 3):
                number += 1
                #if num
                leaf_num = -1
                #tree_sentence.append(root_node)
                root_branch = []
                root_branch_pos = []
                root_coord = []
                root_coord_father = [[0,0], 0, "", 0]
                #root_coord_oneself = []
                #print(root_node)
                #if number > 20:
                    #print(number, root_node)
                #print(root_branch, root_node)
                for ind_root, root in enumerate(root_node):
                    branch = []
                    branch_pos = []
                    bracket = 0
                    #print(root)
                    if len(root) <= 3:      #root==1表示当前节点是"("
                        #branch.append(root[0])
                        root_branch.append(root)
                        root_branch_pos.append(root_node_pos[ind_root])
                        root_coord.append(root_father[ind_root])
                        #branch = []
                    else:
                        #print(root_node_pos)
                        pos_list = []
                        #pos = root_node_pos[ind_root][0].replace("(", "")
                        for ind_leaf, leaf in enumerate(root):
                            ####print(ind_leaf, leaf)
                            #print(ind_leaf, leaf)
                            #pos = root_node_pos[ind_root][0].replace("(", "")
                            if ind_leaf>0 and ind_leaf<len(root)-1:
                                branch.append(leaf.lower())
                                #print(leaf)
                                branch_pos.append(root_node_pos[ind_root][ind_leaf])
                                if leaf == "(":
                                    bracket = bracket+1
                                    pos = root_node_pos[ind_root][ind_leaf].replace("(", "")
                                    pos_list.append(pos)
                                if len(pos_list) > 0:
                                    pos_last = pos_list[-1]
                                if leaf == ")":
                                    bracket = bracket-1
                                    del pos_list[-1]
                                #print(bracket)
                                if bracket == 0:
                                    root_coord_father = [[lay, ind_root], 1, pos_last, 0]
                                    root_coord.append(root_coord_father)
                                    root_branch.append(branch)
                                    root_branch_pos.append(branch_pos)
                                    branch = []
                                    branch_pos = []
                    #print("1", root_branch)
                    #print("1", root_branch)
                    #print(root_coord)
                tree_sentence.append(root_branch)
                tree_coord.append(root_coord)
                #print(tree_sentence)
                #print(tree_coord)
                for leaf in root_branch:
                    if leaf_num < len(leaf):
                        leaf_num = len(leaf)
                root_node = root_branch
                root_node_pos = root_branch_pos
                #print(root_node)
                root_father = root_coord
                lay += 1
            #print(tree_sentence)
            #print(tree_coord)
            tree_save1.append(tree_sentence)
            local_save1.append(tree_coord)
            #for ind, deap in enumerate(root_father):
            for ind_leaf, leaf in enumerate(tree_coord[-1]):
                q=-1
                p=-1
                while(q!=0 or p != 0):
                    #print(tree_coord, leaf)
                    if tree_coord[leaf[0][0]][leaf[0][1]][1] < leaf[1]+1:
                        tree_coord[leaf[0][0]][leaf[0][1]][1] = leaf[1]+1
                    q = leaf[0][0]
                    p = leaf[0][1]
                    leaf = tree_coord[leaf[0][0]][leaf[0][1]]
            for ind_sent,sent in enumerate(tree_sentence):
                for ind_word,word in enumerate(sent):
                    if tree_coord[ind_sent][ind_word][1] < 20:
                        phrase = " ".join(word)
                        phrase = phrase.replace(" )", "")
                        phrase = phrase.replace("( ", "")
                        phrase = phrase.lower()
                        phrasecount[phrase] = phrasecount.get(phrase, 0) + 1
        root_father = [[[0,0], 0, "", 1]]
        root_pos = []
        #print(root_node)
        #print(root_node)
        leaf_num = -1
        tree_sentence = []
        #tree_sentence
        tree_coord = []
        #print(root_node)
        tree_coord.append(root_father)
        tree_sentence.append(root_node2)
        lay = 0
        number = 0
        
        #print(root_node)
        if root_node2!=[[]]:
            while(leaf_num != 3):
                number += 1
                #if num
                leaf_num = -1
                #tree_sentence.append(root_node)
                root_branch = []
                root_branch_pos = []
                root_coord = []
                root_coord_father = [[0,0], 0, "", 0]
                #root_coord_oneself = []
                #print(root_node)
                #if number > 20:
                    #print(number, root_node)
                #print(root_branch, root_node)
                for ind_root, root in enumerate(root_node2):
                    branch = []
                    branch_pos = []
                    bracket = 0
                    #print(root)
                    if len(root) <= 3:      #root==1表示当前节点是"("
                        #branch.append(root[0])
                        root_branch.append(root)
                        root_branch_pos.append(root_node_pos2[ind_root])
                        root_coord.append(root_father[ind_root])
                        #branch = []
                    else:
                        #print(root_node_pos)
                        pos_list = []
                        #pos = root_node_pos[ind_root][0].replace("(", "")
                        for ind_leaf, leaf in enumerate(root):
                            ####print(ind_leaf, leaf)
                            #print(ind_leaf, leaf)
                            #pos = root_node_pos[ind_root][0].replace("(", "")
                            if ind_leaf>0 and ind_leaf<len(root)-1:
                                branch.append(leaf.lower())
                                #print(leaf)
                                branch_pos.append(root_node_pos2[ind_root][ind_leaf])
                                if leaf == "(":
                                    bracket = bracket+1
                                    pos = root_node_pos2[ind_root][ind_leaf].replace("(", "")
                                    pos_list.append(pos)
                                if len(pos_list) > 0:
                                    pos_last = pos_list[-1]
                                if leaf == ")":
                                    bracket = bracket-1
                                    del pos_list[-1]
                                #print(bracket)
                                if bracket == 0:
                                    root_coord_father = [[lay, ind_root], 1, pos_last, 0]
                                    root_coord.append(root_coord_father)
                                    root_branch.append(branch)
                                    root_branch_pos.append(branch_pos)
                                    branch = []
                                    branch_pos = []
                    #print("1", root_branch)
                    #print("1", root_branch)
                    #print(root_coord)
                tree_sentence.append(root_branch)
                tree_coord.append(root_coord)
                #print(tree_sentence)
                #print(tree_coord)
                for leaf in root_branch:
                    if leaf_num < len(leaf):
                        leaf_num = len(leaf)
                root_node2 = root_branch    #root_node
                root_node_pos2 = root_branch_pos
                #print(root_node)

                root_father = root_coord
                lay += 1
            #print(tree_sentence)
            #print(tree_coord)
            tree_save2.append(tree_sentence)
            local_save2.append(tree_coord)
            #for ind, deap in enumerate(root_father):
            for ind_leaf, leaf in enumerate(tree_coord[-1]):
                q=-1
                p=-1
                while(q!=0 or p != 0):
                    #print(tree_coord, leaf)
                    if tree_coord[leaf[0][0]][leaf[0][1]][1] < leaf[1]+1:
                        tree_coord[leaf[0][0]][leaf[0][1]][1] = leaf[1]+1
                    q = leaf[0][0]
                    p = leaf[0][1]
                    leaf = tree_coord[leaf[0][0]][leaf[0][1]]
            for ind_sent,sent in enumerate(tree_sentence):
                for ind_word,word in enumerate(sent):
                    if tree_coord[ind_sent][ind_word][1] < 20:
                        phrase = " ".join(word)
                        phrase = phrase.replace(" )", "")
                        phrase = phrase.replace("( ", "")
                        phrase = phrase.lower()
                        phrasecount[phrase] = phrasecount.get(phrase, 0) + 1
    phrase_word = phrasecount.items()
    phrase_word = sorted(phrase_word, key = lambda x : x[1], reverse = True)    # 利用词频排序
    #phrase_word = phrase_word 
    #print(phrasecount)
    phrase_wordlist = {item[0]: index+1 for index, item in enumerate(phrase_word)}
    data_tatal, data_adjoin_tatal = [], []
    sentence_pair = []
    #print(classes)  
    end_time_line = time.time() 
    print("load line spend time",  end_time_line - start_time)
    for ind_tree,tree in enumerate(tree_save1):    #句子对单独存储
        data = [[],[]]
        #print(filename, "ind_tree", ind_tree)
        data_adjoin = []
        relation = []
        ###for ind_sent,sent in enumerate(tree):
        words1 = tree[-1]
        words2 = tree_save2[ind_tree][-1]
        sentence1 = " ".join(tree[0][0])
        sentence1 = sentence1.replace(" )", "")
        sentence1 = sentence1.replace("( ", "")
        sentence2 = " ".join(tree_save2[ind_tree][0][0])
        sentence2 = sentence2.replace(" )", "")
        sentence2 = sentence2.replace("( ", "")
        sentence1 = sentence1.lower()
        sentence2 = sentence2.lower()
        #print(classes)
        sentence_pair.append([[sentence1, sentence2], class_list[ind_tree]])
        #print([[sentence1, sentence2], class_list[ind_tree]])
        local_tree1 = local_save1[ind_tree]
        local_words1 = local_tree1[-1]
        local_tree2 = local_save2[ind_tree]
        local_words2 = local_tree2[-1]
###print()
        #print(tree)
        #print(local_tree1)
        for indsent,sentences in enumerate(tree):
            if indsent > 0:
                for indph, phrase in enumerate(sentences):
                    if indph + 1 <= len(sentences)-1:
                        phrase = " ".join(phrase)
                        phrase = phrase.replace(" )", "")
                        phrase = phrase.replace("( ", "")
                        phrase_next = " ".join(sentences[indph + 1])
                        phrase_next = phrase_next.replace(" )", "")
                        phrase_next = phrase_next.replace("( ", "")
                        data_adjoin.append([phrase, phrase_next])
        for indsent,sentences in enumerate(tree_save2[ind_tree]):
            if indsent > 0:
                for indph, phrase in enumerate(sentences):
                    if indph + 1 <= len(sentences)-1:
                        phrase = " ".join(phrase)
                        phrase = phrase.replace(" )", "")
                        phrase = phrase.replace("( ", "")
                        phrase_next = " ".join(sentences[indph + 1])
                        phrase_next = phrase_next.replace(" )", "")
                        phrase_next = phrase_next.replace("( ", "")
                        
                        data_adjoin.append([phrase, phrase_next])
        #print(data_adjoin)
        data_adjoin_tatal.append(data_adjoin)
        for ind, word in enumerate(words1):
            #data.append(phrase_wordlist[word[0]])
            #print(local_words1)
            layer = local_words1[ind][1]
            local = [-1, ind]

            while(local_tree1[local[0]][local[1]][3] == 0):
                #local = []
                local_father = local_tree1[local[0]][local[1]][0]
                local_pos = local_tree1[local[0]][local[1]][2]
                local_tree1[local[0]][local[1]][3] = 1
                phrase_local = " ".join(tree[local[0]][local[1]])
                phrase_local_father = " ".join(tree[local_father[0]][local_father[1]])
                phrase_local = phrase_local.replace(" )", "")
                phrase_local = phrase_local.replace("( ", "")
                phrase_local = phrase_local.lower()
                phrase_local_father = phrase_local_father.replace(" )", "")
                phrase_local_father = phrase_local_father.replace("( ", "")
                phrase_local_father = phrase_local_father.lower()
                #print(local_father, phrase_local, phrase_local_father)
                data[0].append([[phrase_local_father, phrase_local], local_pos])
                layer = local_tree1[local_father[0]][local_father[1]][1]
                local = local_father
        for ind, word in enumerate(words2):
            #data.append(phrase_wordlist[word[0]])
            layer = local_words2[ind][1]
            local = [-1, ind]
            while(local_tree2[local[0]][local[1]][3] == 0):
                #local = []
                local_father = local_tree2[local[0]][local[1]][0]
                local_pos = local_tree2[local[0]][local[1]][2]
                local_tree2[local[0]][local[1]][3] = 1
                phrase_local = " ".join(tree_save2[ind_tree][local[0]][local[1]])
                phrase_local = phrase_local.replace(" )", "")
                phrase_local = phrase_local.replace("( ", "")
                phrase_local = phrase_local.lower()
                phrase_local_father = " ".join(tree_save2[ind_tree][local_father[0]][local_father[1]])
                phrase_local_father = phrase_local_father.replace(" )", "")
                phrase_local_father = phrase_local_father.replace("( ", "")
                phrase_local_father = phrase_local_father.lower()
                #print(local_father, phrase_local, phrase_local_father)
                data[1].append([[phrase_local_father, phrase_local], local_pos])
                layer = local_tree2[local_father[0]][local_father[1]][1]
                local = local_father
        #print(data)
        data_tatal.append(data)
    D = []
    #print(len(data_tatal))
    taken_pos_dic, phrase_pos_dic = {}, {}
    phrase_list = []
    phrase_list_pos = []
    end_time_tree = time.time() 
    print("load tree spend time",  end_time_tree - end_time_line)
    print("1232222222222", len(data_tatal))
    for i in range(len(data_tatal)):
        utters = sentence_pair[i]
        #print("i", filename, i)
        #for j in range(len(data[i][1])):
        #print(i, "000000000000000", data_tatal[i][0])
        #print(i, "111111111111111", data_tatal[i][1])
        if utters[0][0] != utters[0][1] and utters[1] != "-" and len(utters[0][0].split(" ")) > 1 and len(utters[0][1].split(" ")) > 1:
            spacy_feats = get_feats(utters, data_tatal[i])
            #print("88888888", len(spacy_feats['phrase_word']))
            #print("99999999", len(spacy_feats['phrase_pos']))
            phrase_list = phrase_list + spacy_feats['phrase_word']
            #print(phrase_list)
            phrase_list_pos = phrase_list_pos + spacy_feats['phrase_pos']
            d = {}
            d['us1'] = utters[0][0]    #表示句子1
            d['us2'] = utters[0][1]    #表示句子2
            d['label'] = utters[1]     #句子之间的标签
            #print(utters[1])
            #d['us'] = utters

            d['feats'] = spacy_feats #其本身是个字典，字典关键字对应的是列表                
            # d['x_type'] = data[i][1][j]["x_type"]
            # d['y_type'] = data[i][1][j]["y_type"]
            # d['rid'] = data[i][1][j]["rid"]
            # d['r'] = data[i][1][j]["r"]
            # d['t'] = data[i][1][j]["t"]
        #print(d['feats']['tokens_pos'])   #phrase_tag    tokens_pos
            # d['x'] = [str(token) for token in nlp(data[i][1][j]["x"])]
            # d['x'] = ''.join(d['x']) if 'Speaker' in d['x'] else ' '.join(d['x'])
            # d['y'] = [str(token) for token in nlp(data[i][1][j]["y"])]
            # d['y'] = ''.join(d['y']) if 'Speaker' in d['y'] else ' '.join(d['y'])
            D.append(d)   #
        #phrases = [oo for o in d['feats']["phrase_before"] for oo in o]
        #print(d['feats']['tokens_sentence'])
            tokens += [oo for o in d['feats']['tokens_sentence'] for oo in o]  #好像没有设置tokens，后面需要什么再给什么吧
            phrases += [o for o in d['feats']['phrase_word']] # for oo in o]    
            sentence_list += [o for o in d['feats']['sent_nophrase']] 
            #d['feats']['phrase']
    #print(taken_pos_dic)
   # for key in taken_pos_dic:
        #print("    " + key)
    #print(phrase_pos_dic)
    #for key in phrase_pos_dic:
       # print("!" + key)
    end_time_data = time.time() 
    print("load data tatal spend time",  end_time_data - end_time_tree)
    return phrases, tokens, sentence_list, D, phrase_list, phrase_list_pos  #返回的中间D是主要
def load_data_c(filename):         #和上面几乎一样，没查出有什么差别
    tokens = []
    word_pairs = {}
    with open(filename) as infile:
        data = json.load(infile)
    D = []
    for i in range(len(data)):
        utters = data[i][0]
        spacy_feats = get_feats(utters, word_pairs)
        for j in range(len(data[i][1])):
            for l in range(1, len(data[i][0])+1):
                d = {}
                d['us'] = utters[:l]
                d['feats'] = {k:v[:l] for k,v in spacy_feats.items()}
                d['x_type'] = data[i][1][j]["x_type"]
                d['y_type'] = data[i][1][j]["y_type"]
                d['rid'] = data[i][1][j]["rid"]
                d['r'] = data[i][1][j]["r"]
                d['t'] = data[i][1][j]["t"]

                #d['x'] = [str(token) for token in nlp(data[i][1][j]["x"])]
                d['x'] = ''.join(d['x']) if 'Speaker' in d['x'] else ' '.join(d['x'])
                #d['y'] = [str(token) for token in nlp(data[i][1][j]["y"])]
                d['y'] = ''.join(d['y']) if 'Speaker' in d['y'] else ' '.join(d['y'])
                D.append(d)
        
        tokens += [oo for o in d['feats']['tokens'] for oo in o]
        
    return tokens, D


def build_vocab(tokens, min_freq):    #这儿可以直接传输字典,这个搞定
    """ build vocab from tokens and glove words. """
    counter = Counter(t for t in tokens)
    # if min_freq > 0, use min_freq, otherwise keep all glove words
    if config.min_freq > 0:    #设置最小频率
        v = sorted([t for t in counter if counter.get(t) >= min_freq], key=counter.get, reverse=True)
    else:
        v = sorted(counter, key=counter.get, reverse=True)
    # add special tokens and entity mask tokens
    #v = constant.VOCAB_PREFIX + v  #应该是一个词典
    print("vocab built with {}/{} words.".format(len(v), len(counter)))
    return v, counter    #counter的计数
def build_phrase(phrase_list):    #这儿可以直接传输字典,这个搞定
    """ build vocab from tokens and glove words. """
    #counter = Counter(t for t in phrase_list)
    phrase_dic, phrase_str_list = [], []
    phrase_dic_dic = {}
    print("phrase_list_len", len(phrase_list))
    for ind, phrase in enumerate(phrase_list):
        phrase_str = " ".join(phrase)
        phrase_dic_dic[phrase_str] = phrase_dic_dic.get(phrase_str, 0) + 1
        #if phrase not in phrase_dic:
        if phrase_dic_dic[phrase_str] == 1:
            phrase_dic.append(phrase)
            phrase_str_list.append(phrase_str)
    one, two, three, four, five, six, seven = 0, 0, 0, 0, 0, 0, 0
    for phrase in phrase_str_list:
        #print(phrase_dic_dic[phrase])
        if phrase_dic_dic[phrase] == 1:
            one = one + 1
        if phrase_dic_dic[phrase] == 2:
            two = two + 1
        if phrase_dic_dic[phrase] == 3:
            three = three + 1
        if phrase_dic_dic[phrase] == 4:
            four = four + 1
        if phrase_dic_dic[phrase] == 5:
            five = five + 1
        if phrase_dic_dic[phrase] == 6:
            six = six + 1
        if phrase_dic_dic[phrase] > 6:
            seven = seven + 1
    print(one, two, three, four, five, six, seven)
    # if min_freq > 0, use min_freq, otherwise keep all glove words
    #v = sorted([t for t in phrase_dic], key=phrase_dic.get, reverse=True)
    # add special tokens and entity mask tokens
    #v = constant.VOCAB_PREFIX + v  #应该是一个词典
    print("vocab built with {}/{} phrase.".format(len(phrase_dic), len(phrase_dic)))
    return phrase_dic, phrase_str_list, phrase_dic_dic     #counter的计数
def build_sentence(sentence_list):    #这儿可以直接传输字典,这个搞定
    """ build vocab from tokens and glove words. """
    sent_dic, sent_str_list = [], []
    sent_dic_dic = {}
    for ind, sent in enumerate(sentence_list):
        sent_str = " ".join(sent)
        sent_dic_dic[sent_str] = sent_dic_dic.get(sent_str, 0) + 1
        #if phrase not in phrase_dic:
        if sent_dic_dic[sent_str] == 1:
        #if sent not in sent_dic:
            sent_dic.append(sent)
            sent_str_list.append(sent_str)
        #sent_dic[sent] = sent_dic.get(sent, 0) + 1
    one, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    for phrase in sent_dic_dic:
        #print(phrase_dic_dic[phrase])
        if sent_dic_dic[phrase] == 1:
            one = one + 1
        if sent_dic_dic[phrase] == 2:
            two = two + 1
        if sent_dic_dic[phrase] == 3:
            three = three + 1
        if sent_dic_dic[phrase] == 4:
            four = four + 1
        if sent_dic_dic[phrase] == 5:
            five = five + 1
        if sent_dic_dic[phrase] == 6:
            six = six + 1
        if sent_dic_dic[phrase] == 7:
            seven = seven + 1
        if sent_dic_dic[phrase] == 8 or sent_dic_dic[phrase] == 9:
            eight = eight + 1
        if sent_dic_dic[phrase] == 10 or sent_dic_dic[phrase] == 11:
            nine = nine + 1
        if sent_dic_dic[phrase] == 12 or sent_dic_dic[phrase] == 13:
            ten = ten + 1
        if sent_dic_dic[phrase] >= 14 and sent_dic_dic[phrase] <= 17:
            eleven = eleven + 1
        if sent_dic_dic[phrase] > 18:
            twelve = twelve + 1
    print("one", one, "two", two, "three", three, "four", four, "five", five, "six", six, "seven", seven, "eight", eight, "nine", nine, "nine", ten, "eleven", eleven, "twelve", twelve)
    #v = sorted([t for t in sent_dic], key=sent_dic.get, reverse=True)
    # add special tokens and entity mask tokens
    #v = constant.VOCAB_PREFIX + v  #应该是一个词典
    print("vocab built with {}/{} sentence.".format(len(sent_dic), len(sent_dic)))
    return sent_dic, sent_str_list    #counter的计数
def count_oov(tokens, vocab):
    c = Counter(t for t in tokens)
    total = sum(c.values())
    matched = sum(c[t] for t in vocab)
    return total, total-matched       #求和
def embed_list(w_list, p_list, s_list):
    start_time = time.time() 
    end_time = time.time()   
    duration = end_time - start_time
    print("emb ceshi", duration)
    embed_dic_sent, embed_dic_phrase, embed_dic_word = {}, {}, {}
    sent_dic = {}
    phrase_dic = {}
    #phrase_list, sent_list, word_list = [], [], []
    f = Bertsent('cls')
    embed_dic_word['<UNK>'] = np.random.uniform(-0.01,0.01,(config.embed_dim))
    start_time = time.time()
    print(start_time)
    w_list_list = [w_list[i:i+1000] for i in range(0,len(w_list), 1000)]
    for word_list in w_list_list:
        out_word = sent2bert(word_list)###print(phrase)
        out_word = out_word.tolist()
        print(111111)
        for ind, word in enumerate(word_list):
            embed_dic_word[word] = out_word[ind]
            #print("7777777", phrase_list)
    print(555555)
    s_list_list = [s_list[i:i+1000] for i in range(0,len(s_list), 1000)]
    for sent in s_list_list:
        #print(sent)
        out_sent = sent2bert(sent)###print(phrase)
    #print(6666666666666)
        out_sent = out_sent.tolist()
    #print(222222)
        for ind, sent_list in enumerate(sent):
            embed_dic_sent[sent_list] = out_sent[ind]
    print(44444444444)
    p_list_list = [p_list[i:i+800] for i in range(0,len(p_list), 800)]
    a = 0
    for phrase in p_list_list:
        a = a+1
        out_phrase = sent2bert(phrase)###print(phrase)
        out_phrase = out_phrase.tolist()
        print(a)
    
        for ind, phrase_split in enumerate(phrase):
            embed_dic_phrase[phrase_split] = out_phrase[ind]
    end_time = time.time() 
    print(len(embed_dic_phrase))
    print("emb time", end_time, end_time - start_time)
    return embed_dic_sent, embed_dic_phrase, embed_dic_word
def embed_list_single(e_list_list):
    #dim_re = nn.Linear(768, embed_dim, bias=False)
    start_time = time.time() 
    end_time = time.time()   
    duration = end_time - start_time
    print("emb ceshi", duration)
    embed_dic = {}
    pca = PCA(n_components=config.embed_dim)
    #phrase_list, sent_list, word_list = [], [], []
    sent2bert = Bertsent('cls')
    embed_dic['<UNK>'] = np.random.uniform(-0.01,0.01,(config.embed_dim))
    start_time = time.time()
    print(start_time)
    embed_str_list = []
    #for str in e_list_list:
        #if str_dic[str] == 1:
            #embed_str_list.append(str)
    embed_list = [e_list_list[i:i+700] for i in range(0,len(e_list_list), 700)]
    if len(embed_list[-1]) < config.embed_dim:
        embed_list[-2] = embed_list[-2] + embed_list[-1]
        del embed_list[-1]
    a = 0
    print("111", len(embed_list))
    for word_list in embed_list:
        #print("222", len(word_list))
        a = a+1
        
        if a%100==0:
            print(a)
        out_word = sent2bert(word_list)###print(phrase)
        print("22222", out_word)
        out_word = pca.fit_transform(out_word.detach().numpy() )
        out_word = out_word.tolist()
        #print()


        for ind, word in enumerate(word_list):
            embed_dic[word] = out_word[ind]
            #print("7777777", phrase_list)
    end_time = time.time() 
    print("emb time", end_time, end_time - start_time)
    return embed_dic
def embed(data):
    start_time = time.time() 
    end_time = time.time()   
    duration = end_time - start_time
    print("emb ceshi", duration)
    embed_dic = {}
    sent_dic = {}
    phrase_dic = {}
    sent2bert = Bertsent('cls')
    embed_dic['<UNK>'] = np.random.uniform(-0.01,0.01,(config.embed_dim))
    start_time = time.time()
    print(start_time)
    for feats_dic in (data):
        
        phrase_word = feats_dic['feats']['phrase_word']
        sent_word = feats_dic['feats']['sent_nophrase']
        phrase_list, sent_list, word_list = [], [], []
        #print(phrase_word)
        if sent_word != []:
            word_list = sent_word[0] + sent_word[1]
            #for sent in :
            #print(word_list)
            out_word = sent2bert(word_list)###print(phrase)
            out_word = out_word.tolist()
            for ind, word in enumerate(word_list):
                embed_dic[word] = out_word[ind]
            for sent in sent_word:
                #print("666666", phrase)
                sent = ' '.join(word for word in sent)
                sent_list.append(sent)
            #print("7777777", phrase_list)
            out_sent = sent2bert(sent_list)###print(phrase)
            out_sent = out_sent.tolist()
            for ind, sent in enumerate(sent_list):
                embed_dic[sent] = out_sent[ind]
        if phrase_word != []:
            for phrase in phrase_word:
                #print("666666", phrase)
                phrase = ' '.join(word for word in phrase)
                phrase_list.append(phrase)
            #print("7777777", phrase_list)
            out_phrase = sent2bert(phrase_list)###print(phrase)
            out_phrase = out_phrase.tolist()
            for ind, phrase in enumerate(phrase_list):
                embed_dic[phrase] = out_phrase[ind]
        #print(len(embed_dic))
    end_time = time.time() 
    print(len(embed_dic))
    print("emb time", end_time, end_time - start_time)
    return embed_dic
            #print("555555", embed_dic)
def cluster(data):
    phrase_word_list = []
    sent_nophrase_list = []
    count_phrase = []
    count_sent = []
    count_a = []
    count_b = []
    count_phrase_b = []
    for feats_dic in (data):
        phrase_word = feats_dic['feats']['phrase_word']
        phrase_word_list.append(phrase_word)
        sent_word = feats_dic['feats']['sent_nophrase']
        sent_nophrase_list.append(sent_word)
    c = 0
    for ind, phrase_word in enumerate(phrase_word_list):
        
        
        for ind1, phrase_word1 in enumerate(phrase_word_list):
            a = 0
            b = 0
            for phrase in phrase_word1:
                if phrase in phrase_word:
                    a = a + 1
            for sent in sent_nophrase_list[ind1]:
                if sent in sent_nophrase_list[ind]:
                    b = b + 1
            if a > 3 and b != 1:
                count_b.append(ind1)
            
                count_a.append([a, b])
            if len(count_b) >= config.batch_size:
                break
            
        if len(count_b) > 1:
            c = c + 1

            print(c, ind, len(count_b), count_b)
        count_phrase_b.append(count_b)
        count_phrase.append(count_a)
        count_a = []
        count_b = []
        count_sent.append(b)
    phrase_dic = {}
    for ind_b, phrase_b in enumerate(count_phrase_b):
        if len(phrase_b) > 1:
            phrase_dic[len(phrase_b)] = phrase_dic.get(len(phrase_b), [])
            phrase_dic[len(phrase_b)].append(phrase_b)
    phrase_dic_keys = sorted(phrase_dic.keys())
    phrase_list_all = []
    for dic_ind in phrase_dic.keys():
        for dic_ind_list in phrase_dic[dic_ind]:
            phrase_list_all = phrase_list_all + dic_ind_list
    phrase_list_all = list(set(phrase_list_all))
    phrase_list_all = sorted(phrase_list_all)
    print("phrase_list_all", len(phrase_list_all))
    phrase_alone = []
    for i in range(0, len(data)):
        if i not in phrase_list_all:
            phrase_alone.append(data[i])
            
    for dic_i in phrase_dic_keys:
        for ind_dic_list, phrase_list in enumerate(phrase_dic[dic_i]):
            for ind_dic, phrase in enumerate(phrase_list):
                phrase_alone.append(data[ind_dic])
        print(dic_i, len(phrase_alone))
    return phrase_alone
def load_dataset():
    
    if os.path.exists(config.proce_f_word) and os.path.exists(config.proce_f_phrase) and os.path.exists(config.proce_f_sentence): ##os.path.exists(config.proce_f)  ata + test_data)
        # embed_d
        print("LOADING dialogre dataset")
        with open(config.proce_f, "rb") as f: [train_data, dev_data, test_data, vocab, phrase_list_repet, phrase_list_pos_repet, vocab_ph, vocab_sent, v, s_list, p, phrase_dic_dic] = pickle.load(f)
            # a = 1
        star_time = time.time()
        with open(config.proce_f_word, "rb") as f: [embed_dic_word] = pickle.load(f)
        time_star = time.time()
        print("load word time",  time_star - star_time)
        with open(config.proce_f_phrase, "rb") as f: [embed_dic_phrase] = pickle.load(f)
        end_time = time.time()
        print("pichle load time", end_time - time_star)
        with open(config.proce_f_sentence, "rb") as f: [embed_dic_sent] = pickle.load(f)

        return train_data, dev_data, test_data, vocab, phrase_list_repet, phrase_list_pos_repet, vocab_ph, vocab_sent, v, s_list, p, embed_dic_sent, embed_dic_phrase, embed_dic_word, phrase_dic_dic

    print("Preprocessing dialogre dataset... (This can take some time)")
    # load files
    print("loading files...", config.train1_f, config.train2_f, config.train3_f, config.train4_f, config.train5_f, config.train6_f)
    if os.path.exists(config.w_p_s_list):
        print("LOADING dialogre dataset")
        with open(config.w_p_s_list, "rb") as f: [train_phrases, train_tokens, train_sentence_list, train_data, phrase_list_tr, phrase_list_pos_tr, dev_phrases, dev_tokens, dev_sentence_list, dev_data, phrase_list_dev, phrase_list_pos_dev, test_phrases, test_tokens, test_sentence_list, test_data, phrase_list_ts, phrase_list_pos_ts] = pickle.load(f)
    else:
        
        train1_phrases, train1_tokens, train1_sentence_list, train1_data, phrase1_list_tr, phrase1_list_pos_tr = load_data(config.train1_f)
        train2_phrases, train2_tokens, train2_sentence_list, train2_data, phrase2_list_tr, phrase2_list_pos_tr = load_data(config.train2_f)
        train3_phrases, train3_tokens, train3_sentence_list, train3_data, phrase3_list_tr, phrase3_list_pos_tr = load_data(config.train3_f)
        train4_phrases, train4_tokens, train4_sentence_list, train4_data, phrase4_list_tr, phrase4_list_pos_tr = load_data(config.train4_f)
        train5_phrases, train5_tokens, train5_sentence_list, train5_data, phrase5_list_tr, phrase5_list_pos_tr = load_data(config.train5_f)
        train6_phrases, train6_tokens, train6_sentence_list, train6_data, phrase6_list_tr, phrase6_list_pos_tr = load_data(config.train6_f)
        train_data_1 = train1_data + train2_data + train3_data
        train_phrases = train1_phrases + train2_phrases + train3_phrases + train4_phrases + train5_phrases + train6_phrases
        train_tokens = train1_tokens + train2_tokens + train3_tokens + train4_tokens + train5_tokens + train6_tokens
        train_sentence_list = train1_sentence_list + train2_sentence_list + train3_sentence_list +train4_sentence_list + train5_sentence_list + train6_sentence_list
        train_data = train1_data + train2_data + train3_data + train4_data + train5_data + train6_data
        phrase_list_tr = phrase1_list_tr + phrase2_list_tr + phrase3_list_tr + phrase4_list_tr + phrase5_list_tr + phrase6_list_tr
        phrase_list_pos_tr = phrase1_list_pos_tr + phrase2_list_pos_tr + phrase3_list_pos_tr +phrase4_list_pos_tr + phrase5_list_pos_tr + phrase6_list_pos_tr
        dev_phrases, dev_tokens, dev_sentence_list, dev_data, phrase_list_dev, phrase_list_pos_dev = load_data(config.val_f)
        test_phrases, test_tokens, test_sentence_list, test_data, phrase_list_ts, phrase_list_pos_ts = load_data(config.test_f)
        with open(config.w_p_s_list, 'wb') as outfile:
            pickle.dump([train_phrases, train_tokens, train_sentence_list, train_data, phrase_list_tr, phrase_list_pos_tr, dev_phrases, dev_tokens, dev_sentence_list, dev_data, phrase_list_dev, phrase_list_pos_dev, test_phrases, test_tokens, test_sentence_list, test_data, phrase_list_ts, phrase_list_pos_ts], outfile)
    start_time = time.time()
    phrase_list_repet = []#phrase_list_tr
    phrase_list_pos_repet = []#phrase_list_pos_tr
    phrase_list_tr_dic = {}
    #phrase_list_pos_repet_dic = {}
    # for ind, phrase in enumerate(phrase_list_tr):
        # if phrase not in phrase_list_repet:
            # phrase_list_repet.append(phrase)
            # phrase_list_pos_repet.append(phrase_list_pos_tr[ind])
    
    end_time = time.time()
    print("phrase_pos", end_time - start_time)
    phrase_list_repet_dic = []#phrase_list_tr
    phrase_list_pos_repet_dic = []#phrase_list_pos_tr
    for ind, phrase in enumerate(phrase_list_tr):
        phrase_str = ' '.join(word for word in phrase)
        #print(phrase_str)
        phrase_list_tr_dic[phrase_str] = phrase_list_tr_dic.get(phrase_str, 0) + 1
        #if phrase not in phrase_list_repet:
        if phrase_list_tr_dic[phrase_str] == 1:
            #print("11111111111")
            phrase_list_repet.append(phrase)
            phrase_list_pos_repet.append(phrase_list_pos_tr[ind])
    end_time_dic = time.time()
    print("phrase_pos_dic", end_time_dic - end_time)
    # phrase_list_random = random.sample(phrase_list_repet, 10)
    #print("12345678098765432", phrase_list_random)
    # load glove
    print("loading glove...")
    #glove_vocab = load_glove_vocab(config.glove_f, config.embed_dim)
    #print("{} words loaded from glove.".format(len(glove_vocab)))
    
    print("building vocab...")

    # TODO: The vocab should contain all 3 splits? 
    #print(train_phrases)
    p, p_str_list, phrase_dic_dic = build_phrase(train_phrases + dev_phrases + test_phrases)
    v, v_counter = build_vocab(train_tokens + dev_tokens + test_tokens, config.min_freq)
    #print(train_phrases)

    s_list, s_str_list = build_sentence(train_sentence_list + dev_sentence_list + test_sentence_list)
    print("calculating oov...")
    datasets = {'train': train_tokens, 'dev': dev_tokens, 'test': test_tokens}
    datasets_phrase = {'train_phrase': train_phrases, 'dev_phrase': dev_phrases, 'test_phrase': test_phrases}
    for dname, d in datasets.items():
        total, oov = count_oov(d, v)
        print("{} oov: {}/{} ({:.2f}%)".format(dname, oov, total, oov*100.0/total))
    print("building embeddings...")
    #embedding = build_embedding(config.glove_f, v, config.embed_dim)
    #embedding_ph = build_embedding(config.glove_f, p, config.embed_dim)
    #print("embedding size: {} x {}".format(*embedding.shape))
    #print("embedding_ph size: {} x {}".format(*embedding_ph.shape))
    print("dumping to files...")
    #sentence_vocab = Vocab_sent(s_list, s_counter)
    #vocab_ph = Vocab(p, p_counter)
    with open(config.proce_f, 'wb') as outfile:
        vocab = Vocab(v, v_counter)
        vocab_sent = Vocab_sent(s_list, p, v)
        vocab_ph = Vocab_phrase(p, v)
        pickle.dump([train_data, dev_data, test_data, vocab, phrase_list_repet, phrase_list_pos_repet, vocab_ph, vocab_sent, v, s_list, p, phrase_dic_dic], outfile)
    embed_dic_word = embed_list_single(v)
    with open(config.proce_f_word, 'wb') as outfile:
        pickle.dump([embed_dic_word], outfile)
    embed_dic_phrase = embed_list_single(p_str_list)
    with open(config.proce_f_phrase, 'wb') as outfile:
        pickle.dump([embed_dic_phrase], outfile)
    embed_dic_sent = embed_list_single(s_str_list)
    with open(config.proce_f_sentence, 'wb') as outfile:
        pickle.dump([embed_dic_sent], outfile)
    #np.save(config.embed_f, embedding)
    print("all done.")        
    
    #embed_dic = embed(train_data + dev_data + test_data)   #enmbed_bert = 
    #train_data = cluster(train_data)
    #dev_data = cluster(dev_data)
    return train_data, dev_data, test_data, vocab, phrase_list_repet, phrase_list_pos_repet, vocab_ph, vocab_sent, v, s_list, p, embed_dic_sent, embed_dic_phrase, embed_dic_word, phrase_dic_dic

def load_dataset_c():
    if os.path.exists(config.proce_f_c):
        print("LOADING dialogre dataset for conversation setting")
        with open(config.proce_f_c, "rb") as f: [dev_data, test_data] = pickle.load(f)
        return dev_data, test_data

    print("Preprocessing dialogre dataset... (This can take some time)")
    # load files
    print("loading files...")
    dev_phrases, dev_tokens, dev_sentence_list, dev_data, phrase_list_dev, phrase_list_pos_dev = load_data(config.val_f)#load_data_c(config.val_f)
    test_phrases, test_tokens, test_sentence_list, test_data, phrase_list_ts, phrase_list_pos_ts = load_data(config.test_f)#load_data_c(config.test_f)

    print("dumping to files...")
    with open(config.proce_f_c, 'wb') as outfile:
        pickle.dump([dev_data, test_data], outfile)
    print("all done.")

    return dev_data, test_data

def get_original_data(fn):
    tokens = []  
    phrases = []
    class_list = []
    word_pairs = {}
    file = open(fn, "r", encoding = "utf-8")   #路径改一下
    lines = file.readlines()
    #random.shuffle(lines)
    wordcount = {}
    phrasecount = {}
    sentencecount = {}
    tree_save1, tree_save2 = [], []
    local_save1, local_save2 = [], []
    for ind, line in enumerate(lines):
        s = line.strip().split("	")
        classes = s[0]
        #print(classes)
        sent1 = s[3]
        sent2 = s[4]
        sent1_pos, sent2_pos = [], []
        class_list.append(classes)
        #print(sent1)
        #if sentencecount.get(sent1, 0) == 0:
        sentencecount[sent1] = sentencecount.get(sent1, 0) + 1
        sentencecount[sent2] = sentencecount.get(sent2, 0) + 1
        sent1 = sent1.replace(")", " )")
        sent2 = sent2.replace(")", " )")
        #sent1 = sent1.replace("))", ") )")
        #
        sent1 = sent1.split(" ")
        sent2 = sent2.split(" ")
        prior_ind = 0
        ind_wipe = []
        sent1_pos = sent1_pos + sent1
        sent2_pos = sent2_pos + sent2
        #sent1 = sent1.lower()

        for ind, word in enumerate(sent1):
            if word.find("(") != -1:     #说明该字符串中包含"("
                ind_left = ind
                #if ind < len(sent1)-1 and sent1[ind+1].find("(") != -1:     #表示下一个字符串中也包含"("
                sent1[ind] = "("#sent1[ind][0]      #把包含"("的字符串转化为"("
                    #print(sent1[ind])     #这儿把所有标注都去掉不适合当前语料
                #else:
                    #ind_wipe.append(ind)  #否则把当前节点除去，说明当前节点是单词的词性标注加"("
            #if word.find(")") != -1 and word != ")":
                #sent1[ind] = sent1[ind].replace(")", "")      #排除干扰项
                
        #print(sent1)
        #for ind, i in enumerate(ind_wipe):
            #del sent1[ind_wipe[len(ind_wipe)-1 - ind]]
            #prior_ind = ind_left
        #print(sent1)
        for ind, word in enumerate(sent2):
            if word.find("(") != -1:     #说明该字符串中包含"("
                ind_left = ind
                #if ind < len(sent1)-1 and sent1[ind+1].find("(") != -1:     #表示下一个字符串中也包含"("
                sent2[ind] = "("
        root_node, root_node2 = [], []
        root_node_pos, root_node_pos2 = [], []
        root_node.append(sent1[1:len(sent1)-1])
        root_node2.append(sent2[1:len(sent2)-1])
        #print("0", root_node)
        root_node_pos.append(sent1_pos[1:len(sent1_pos)-1])
        root_node_pos2.append(sent2_pos[1:len(sent2_pos)-1])
        root_father = [[[0,0], 0, "", 1]]
        root_pos = []
        #print(root_node)
        #print(root_node)
        leaf_num = -1
        tree_sentence = []
        #tree_sentence
        tree_coord = []
        #print(root_node)
        tree_coord.append(root_father)
        tree_sentence.append(root_node)
        lay = 0
        number = 0
        
        #print(root_node)
        if root_node!=[[]]:
            while(leaf_num != 3):
                number += 1
                #if num
                leaf_num = -1
                #tree_sentence.append(root_node)
                root_branch = []
                root_branch_pos = []
                root_coord = []
                root_coord_father = [[0,0], 0, "", 0]
                #root_coord_oneself = []
                #print(root_node)
                #if number > 20:
                    #print(number, root_node)
                #print(root_branch, root_node)
                for ind_root, root in enumerate(root_node):
                    branch = []
                    branch_pos = []
                    bracket = 0
                    #print(root)
                    if len(root) <= 3:      #root==1表示当前节点是"("
                        #branch.append(root[0])
                        root_branch.append(root)
                        root_branch_pos.append(root_node_pos[ind_root])
                        root_coord.append(root_father[ind_root])
                        #branch = []
                    else:
                        #print(root_node_pos)
                        pos_list = []
                        #pos = root_node_pos[ind_root][0].replace("(", "")
                        for ind_leaf, leaf in enumerate(root):
                            ####print(ind_leaf, leaf)
                            #print(ind_leaf, leaf)
                            #pos = root_node_pos[ind_root][0].replace("(", "")
                            if ind_leaf>0 and ind_leaf<len(root)-1:
                                branch.append(leaf.lower())
                                #print(leaf)
                                branch_pos.append(root_node_pos[ind_root][ind_leaf])
                                if leaf == "(":
                                    bracket = bracket+1
                                    pos = root_node_pos[ind_root][ind_leaf].replace("(", "")
                                    pos_list.append(pos)
                                if len(pos_list) > 0:
                                    pos_last = pos_list[-1]
                                if leaf == ")":
                                    bracket = bracket-1
                                    del pos_list[-1]
                                #print(bracket)
                                if bracket == 0:
                                    root_coord_father = [[lay, ind_root], 1, pos_last, 0]
                                    root_coord.append(root_coord_father)
                                    root_branch.append(branch)
                                    root_branch_pos.append(branch_pos)
                                    branch = []
                                    branch_pos = []
                    #print("1", root_branch)
                    #print("1", root_branch)
                    #print(root_coord)
                tree_sentence.append(root_branch)
                tree_coord.append(root_coord)
                #print(tree_sentence)
                #print(tree_coord)
                for leaf in root_branch:
                    if leaf_num < len(leaf):
                        leaf_num = len(leaf)
                root_node = root_branch
                root_node_pos = root_branch_pos
                #print(root_node)
                root_father = root_coord
                lay += 1
            #print(tree_sentence)
            #print(tree_coord)
            tree_save1.append(tree_sentence)
            local_save1.append(tree_coord)
            #for ind, deap in enumerate(root_father):
            for ind_leaf, leaf in enumerate(tree_coord[-1]):
                q=-1
                p=-1
                while(q!=0 or p != 0):
                    #print(tree_coord, leaf)
                    if tree_coord[leaf[0][0]][leaf[0][1]][1] < leaf[1]+1:
                        tree_coord[leaf[0][0]][leaf[0][1]][1] = leaf[1]+1
                    q = leaf[0][0]
                    p = leaf[0][1]
                    leaf = tree_coord[leaf[0][0]][leaf[0][1]]
            for ind_sent,sent in enumerate(tree_sentence):
                for ind_word,word in enumerate(sent):
                    if tree_coord[ind_sent][ind_word][1] < 20:
                        phrase = " ".join(word)
                        phrase = phrase.replace(" )", "")
                        phrase = phrase.replace("( ", "")
                        phrase = phrase.lower()
                        phrasecount[phrase] = phrasecount.get(phrase, 0) + 1
        root_father = [[[0,0], 0, "", 1]]
        root_pos = []
        #print(root_node)
        #print(root_node)
        leaf_num = -1
        tree_sentence = []
        #tree_sentence
        tree_coord = []
        #print(root_node)
        tree_coord.append(root_father)
        tree_sentence.append(root_node2)
        lay = 0
        number = 0
        
        #print(root_node)
        if root_node2!=[[]]:
            while(leaf_num != 3):
                number += 1
                #if num
                leaf_num = -1
                #tree_sentence.append(root_node)
                root_branch = []
                root_branch_pos = []
                root_coord = []
                root_coord_father = [[0,0], 0, "", 0]
                #root_coord_oneself = []
                #print(root_node)
                #if number > 20:
                    #print(number, root_node)
                #print(root_branch, root_node)
                for ind_root, root in enumerate(root_node2):
                    branch = []
                    branch_pos = []
                    bracket = 0
                    #print(root)
                    if len(root) <= 3:      #root==1表示当前节点是"("
                        #branch.append(root[0])
                        root_branch.append(root)
                        root_branch_pos.append(root_node_pos2[ind_root])
                        root_coord.append(root_father[ind_root])
                        #branch = []
                    else:
                        #print(root_node_pos)
                        pos_list = []
                        #pos = root_node_pos[ind_root][0].replace("(", "")
                        for ind_leaf, leaf in enumerate(root):
                            ####print(ind_leaf, leaf)
                            #print(ind_leaf, leaf)
                            #pos = root_node_pos[ind_root][0].replace("(", "")
                            if ind_leaf>0 and ind_leaf<len(root)-1:
                                branch.append(leaf.lower())
                                #print(leaf)
                                branch_pos.append(root_node_pos2[ind_root][ind_leaf])
                                if leaf == "(":
                                    bracket = bracket+1
                                    pos = root_node_pos2[ind_root][ind_leaf].replace("(", "")
                                    pos_list.append(pos)
                                if len(pos_list) > 0:
                                    pos_last = pos_list[-1]
                                if leaf == ")":
                                    bracket = bracket-1
                                    del pos_list[-1]
                                #print(bracket)
                                if bracket == 0:
                                    root_coord_father = [[lay, ind_root], 1, pos_last, 0]
                                    root_coord.append(root_coord_father)
                                    root_branch.append(branch)
                                    root_branch_pos.append(branch_pos)
                                    branch = []
                                    branch_pos = []
                    #print("1", root_branch)
                    #print("1", root_branch)
                    #print(root_coord)
                tree_sentence.append(root_branch)
                tree_coord.append(root_coord)
                #print(tree_sentence)
                #print(tree_coord)
                for leaf in root_branch:
                    if leaf_num < len(leaf):
                        leaf_num = len(leaf)
                root_node2 = root_branch    #root_node
                root_node_pos2 = root_branch_pos
                #print(root_node)

                root_father = root_coord
                lay += 1
            #print(tree_sentence)
            #print(tree_coord)
            tree_save2.append(tree_sentence)
            local_save2.append(tree_coord)
            #for ind, deap in enumerate(root_father):
            for ind_leaf, leaf in enumerate(tree_coord[-1]):
                q=-1
                p=-1
                while(q!=0 or p != 0):
                    #print(tree_coord, leaf)
                    if tree_coord[leaf[0][0]][leaf[0][1]][1] < leaf[1]+1:
                        tree_coord[leaf[0][0]][leaf[0][1]][1] = leaf[1]+1
                    q = leaf[0][0]
                    p = leaf[0][1]
                    leaf = tree_coord[leaf[0][0]][leaf[0][1]]
            for ind_sent,sent in enumerate(tree_sentence):
                for ind_word,word in enumerate(sent):
                    if tree_coord[ind_sent][ind_word][1] < 20:
                        phrase = " ".join(word)
                        phrase = phrase.replace(" )", "")
                        phrase = phrase.replace("( ", "")
                        phrase = phrase.lower()
                        phrasecount[phrase] = phrasecount.get(phrase, 0) + 1
    phrase_word = phrasecount.items()
    phrase_word = sorted(phrase_word, key = lambda x : x[1], reverse = True)    # 利用词频排序
    #phrase_word = phrase_word 
    #print(phrasecount)
    phrase_wordlist = {item[0]: index+1 for index, item in enumerate(phrase_word)}
    data_tatal, data_adjoin_tatal = [], []
    sentence_pair = []
    #print(classes)    
    for ind_tree,tree in enumerate(tree_save1):    #句子对单独存储
        data = [[],[]]
        data_adjoin = []
        relation = []
        ###for ind_sent,sent in enumerate(tree):
        words1 = tree[-1]
        words2 = tree_save2[ind_tree][-1]
        sentence1 = " ".join(tree[0][0])
        sentence1 = sentence1.replace(" )", "")
        sentence1 = sentence1.replace("( ", "")
        sentence2 = " ".join(tree_save2[ind_tree][0][0])
        sentence2 = sentence2.replace(" )", "")
        sentence2 = sentence2.replace("( ", "")
        sentence1 = sentence1.lower()
        sentence2 = sentence2.lower()
        #print(classes)
        sentence_pair.append([[sentence1, sentence2], class_list[ind_tree]])
        #print([[sentence1, sentence2], class_list[ind_tree]])
        local_tree1 = local_save1[ind_tree]
        local_words1 = local_tree1[-1]
        local_tree2 = local_save2[ind_tree]
        local_words2 = local_tree2[-1]
###print()
        #print(tree)
        #print(local_tree1)
        for indsent,sentences in enumerate(tree):
            if indsent > 0:
                for indph, phrase in enumerate(sentences):
                    if indph + 1 <= len(sentences)-1:
                        phrase = " ".join(phrase)
                        phrase = phrase.replace(" )", "")
                        phrase = phrase.replace("( ", "")
                        phrase_next = " ".join(sentences[indph + 1])
                        phrase_next = phrase_next.replace(" )", "")
                        phrase_next = phrase_next.replace("( ", "")
                        data_adjoin.append([phrase, phrase_next])
        for indsent,sentences in enumerate(tree_save2[ind_tree]):
            if indsent > 0:
                for indph, phrase in enumerate(sentences):
                    if indph + 1 <= len(sentences)-1:
                        phrase = " ".join(phrase)
                        phrase = phrase.replace(" )", "")
                        phrase = phrase.replace("( ", "")
                        phrase_next = " ".join(sentences[indph + 1])
                        phrase_next = phrase_next.replace(" )", "")
                        phrase_next = phrase_next.replace("( ", "")
                        
                        data_adjoin.append([phrase, phrase_next])
        #print(data_adjoin)
        data_adjoin_tatal.append(data_adjoin)
        for ind, word in enumerate(words1):
            #data.append(phrase_wordlist[word[0]])
            #print(local_words1)
            layer = local_words1[ind][1]
            local = [-1, ind]

            while(local_tree1[local[0]][local[1]][3] == 0):
                #local = []
                local_father = local_tree1[local[0]][local[1]][0]
                local_pos = local_tree1[local[0]][local[1]][2]
                local_tree1[local[0]][local[1]][3] = 1
                phrase_local = " ".join(tree[local[0]][local[1]])
                phrase_local_father = " ".join(tree[local_father[0]][local_father[1]])
                phrase_local = phrase_local.replace(" )", "")
                phrase_local = phrase_local.replace("( ", "")
                phrase_local = phrase_local.lower()
                phrase_local_father = phrase_local_father.replace(" )", "")
                phrase_local_father = phrase_local_father.replace("( ", "")
                phrase_local_father = phrase_local_father.lower()
                #print(local_father, phrase_local, phrase_local_father)
                data[0].append([[phrase_local_father, phrase_local], local_pos])
                layer = local_tree1[local_father[0]][local_father[1]][1]
                local = local_father
        for ind, word in enumerate(words2):
            #data.append(phrase_wordlist[word[0]])
            layer = local_words2[ind][1]
            local = [-1, ind]
            while(local_tree2[local[0]][local[1]][3] == 0):
                #local = []
                local_father = local_tree2[local[0]][local[1]][0]
                local_pos = local_tree2[local[0]][local[1]][2]
                local_tree2[local[0]][local[1]][3] = 1
                phrase_local = " ".join(tree_save2[ind_tree][local[0]][local[1]])
                phrase_local = phrase_local.replace(" )", "")
                phrase_local = phrase_local.replace("( ", "")
                phrase_local = phrase_local.lower()
                phrase_local_father = " ".join(tree_save2[ind_tree][local_father[0]][local_father[1]])
                phrase_local_father = phrase_local_father.replace(" )", "")
                phrase_local_father = phrase_local_father.replace("( ", "")
                phrase_local_father = phrase_local_father.lower()
                #print(local_father, phrase_local, phrase_local_father)
                data[1].append([[phrase_local_father, phrase_local], local_pos])
                layer = local_tree2[local_father[0]][local_father[1]][1]
                local = local_father
        #print(data)
        data_tatal.append(data)
    data = []
    for i in range(len(data_tatal)):
        utters = sentence_pair[i]

        #for j in range(len(data[i][1])):
        if utters[0][0] != utters[0][1] and utters[1] != "-" and len(utters[0][0].split(" ")) > 1 and len(utters[0][1].split(" ")) > 1:
            spacy_feats = get_feats(utters, data_tatal[i])
            d = {}
            data.append(d)
            #d = {}
            data[i]['rid'] = utters[1]     #句子之间的标签
            labels_list = ['neutral', 'contradiction', 'entailment']
            #for i in range(len(data)):
        #for j in range(len([data[i][1]])):
            a = 0
            for ind, t in enumerate(labels_list):
                if t == data[i]['rid']:
                    a = ind
            for k in range(len(data['rid'])):
            #data[i][1][j]["rid"][k] -= 1
                data[i]["rid"][k] = a
    return data

if __name__ == "__main__":
    trn_data, val_data, tst_data, vocab, phrase_list_repet, phrase_list_pos_repet, vocab_ph, vocab_sent, v_list, s_list, p_list, embed_dic_sent, embed_dic_phrase, embed_dic_word  = load_dataset()
    print(aaaaaa)
    val_data_c, test_data_c = load_dataset_c()
    #load_data_c()
