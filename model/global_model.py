import numpy as np
import torch
import torch
import torch.nn as nn
import dgl
from utils.config import config
from model.gat import WSWGAT
from model.bert import Bertsent
from utils import constant
# from transformers import BertTokenizer, BertModel

class ConvEncoder(nn.Module):
    """
    without sent2sent and add residual connection增加残差网络
    adapted from brxx122/hetersumgraph/higraph.py
    """
    def __init__(self, config, vocab, vocab_ph, vocab_sent, embed_dic_sent, embed_dic_phrase, embed_dic_word, phrase_dic_dic):
        super().__init__()
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  #bert-base-uncased
        # self.model = BertModel.from_pretrained('bert-base-uncased')
        self.config = config
        self.vocab = vocab
        self.vocab_ph = vocab_ph
        self.embed_dic_sent = embed_dic_sent
        self.embed_dic_phrase = embed_dic_phrase
        self.embed_dic_word = embed_dic_word
        self.phrase_dic_dic = phrase_dic_dic
        self.vocab_sent = vocab_sent

        self.ws_embed = nn.Embedding(len(constant.pos_i2s), config.edge_embed_size) # bucket = 10
        self.wn_embed = nn.Embedding(config.wn_edge_bucket, config.edge_embed_size) # bucket = 10
        #####self.sent_feature_proj = nn.Linear(900, config.ggcn_hidden_size, bias=False)    #config.glstm_hidden_dim*2
        #self.ner_feature_proj = nn.Linear(config.ner_embed_dim, config.ggcn_hidden_size, bias=False)
        self.glstm = nn.LSTM(self.config.gcn_lin_dim, 
                                self.config.glstm_hidden_dim, 
                                num_layers=self.config.glstm_layers, dropout=0.1,
                                batch_first=True, bidirectional=True)


        # word -> sent
        self.word2sent = WSWGAT(in_dim=config.embed_dim,
                                out_dim=config.ggcn_hidden_size,
                                num_heads=config.word2sent_n_head,
                                attn_drop_out=config.atten_dropout_prob,
                                ffn_inner_hidden_size=config.ffn_inner_hidden_size,
                                ffn_drop_out=config.ffn_dropout_prob,
                                feat_embed_size=config.edge_embed_size, # for edge
                                layerType="W2S"
                                )
        self.word2phrase = WSWGAT(in_dim=config.embed_dim,
                                out_dim=config.ggcn_hidden_size,
                                num_heads=config.word2sent_n_head,
                                attn_drop_out=config.atten_dropout_prob,
                                ffn_inner_hidden_size=config.ffn_inner_hidden_size,
                                ffn_drop_out=config.ffn_dropout_prob,
                                feat_embed_size=config.edge_embed_size, # for edge
                                layerType="W2P"
                                )

        # sent -> word
        self.sent2word = WSWGAT(in_dim=config.ggcn_hidden_size,
                                out_dim=config.embed_dim,
                                num_heads=config.sent2word_n_head, # make sure word_embedding divisible by this number 
                                attn_drop_out=config.atten_dropout_prob,
                                ffn_inner_hidden_size=config.ffn_inner_hidden_size,
                                ffn_drop_out=config.ffn_dropout_prob,
                                feat_embed_size=config.edge_embed_size, # for edge
                                layerType="S2W"
                                )
        self.phrase2word = WSWGAT(in_dim=config.ggcn_hidden_size,
                                out_dim=config.embed_dim,
                                num_heads=config.sent2word_n_head, # make sure word_embedding divisible by this number 
                                attn_drop_out=config.atten_dropout_prob,
                                ffn_inner_hidden_size=config.ffn_inner_hidden_size,
                                ffn_drop_out=config.ffn_dropout_prob,
                                feat_embed_size=config.edge_embed_size, # for edge
                                layerType="P2W"
                                )
        self.phrase2phrasephrase = WSWGAT(in_dim=config.ggcn_hidden_size,
                                out_dim=config.ggcn_hidden_size,
                                num_heads=config.sent2word_n_head, # make sure word_embedding divisible by this number 
                                attn_drop_out=config.atten_dropout_prob,
                                ffn_inner_hidden_size=config.ffn_inner_hidden_size,
                                ffn_drop_out=config.ffn_dropout_prob,
                                feat_embed_size=config.edge_embed_size, # for edge
                                layerType="P2PP"
                                )
        self.phrasephrase2phrase = WSWGAT(in_dim=config.ggcn_hidden_size,
                                out_dim=config.ggcn_hidden_size,
                                num_heads=config.sent2word_n_head, # make sure word_embedding divisible by this number 
                                attn_drop_out=config.atten_dropout_prob,
                                ffn_inner_hidden_size=config.ffn_inner_hidden_size,
                                ffn_drop_out=config.ffn_dropout_prob,
                                feat_embed_size=config.edge_embed_size, # for edge
                                layerType="PP2P"
                                )
        self.phrase2sent = WSWGAT(in_dim=config.ggcn_hidden_size,
                                out_dim=config.ggcn_hidden_size,
                                num_heads=config.word2sent_n_head, # make sure word_embedding divisible by this number 
                                attn_drop_out=config.atten_dropout_prob,
                                ffn_inner_hidden_size=config.ffn_inner_hidden_size,
                                ffn_drop_out=config.ffn_dropout_prob,
                                feat_embed_size=config.edge_embed_size, # for edge
                                layerType="P2S"
                                )
        self.sent2phrase = WSWGAT(in_dim=config.ggcn_hidden_size,
                                out_dim=config.ggcn_hidden_size,
                                num_heads=config.sent2word_n_head, # make sure word_embedding divisible by this number 
                                attn_drop_out=config.atten_dropout_prob,
                                ffn_inner_hidden_size=config.ffn_inner_hidden_size,
                                ffn_drop_out=config.ffn_dropout_prob,
                                feat_embed_size=config.edge_embed_size, # for edge
                                layerType="S2P"
                                )    
        self.phrase2adjoin = WSWGAT(in_dim=config.ggcn_hidden_size,
                                out_dim=config.ggcn_hidden_size,
                                num_heads=config.sent2word_n_head, # make sure word_embedding divisible by this number 
                                attn_drop_out=config.atten_dropout_prob,
                                ffn_inner_hidden_size=config.ffn_inner_hidden_size,
                                ffn_drop_out=config.ffn_dropout_prob,
                                feat_embed_size=config.edge_embed_size, # for edge
                                layerType="P2A"
                                )
        self.negative2phrase = WSWGAT(in_dim=config.ggcn_hidden_size,
                                out_dim=config.ggcn_hidden_size,
                                num_heads=config.sent2word_n_head, # make sure word_embedding divisible by this number 
                                attn_drop_out=config.atten_dropout_prob,
                                ffn_inner_hidden_size=config.ffn_inner_hidden_size,
                                ffn_drop_out=config.ffn_dropout_prob,
                                feat_embed_size=config.edge_embed_size, # for edge
                                layerType="N2P"
                                )

        # node classificat
        # node classification
        self.wh = nn.Linear(config.ggcn_hidden_size, 2)

    def forward(self, graph, dict_id, batch):   #, graph1
        supernode_id = graph.filter_nodes(lambda nodes: nodes.data['unit'] == 1) # supernodes contains sentNode and phraseNode

        # Initialize states
        ####self.set_wordNode_feature(graph, graph1, local_feature_w)
        ##self.set_speakerNode_feature(graph)
        #self.set_argNode_feature(graph)
        self.set_wordSentEdge_feature(graph)
        self.set_SentphraseEdge_feature(graph)
        ####self.set_PhraseNode_feature(graph, graph1, batch, local_feature_p) 
        ####self.set_sentNode_feature(graph, graph1,  batch, local_feature_s)    # [snode, glstm_hidden_dim] -> [snode, n_hidden_size]
        self.set_wordSentEdge_feature(graph)
        self.set_wordPhraseEdge_feature(graph)
        self.set_SentphraseEdge_feature(graph)
        self.set_phrasePhraseEdge_feature(graph)
        #self.set_wordNerEdge_feature(graph)
        #self.set_nerNode_feature(graph)

        wnode_id = graph.filter_nodes(lambda nodes: nodes.data["unit"]==0) # both word node and speaker node
        # the start state
        pnode_id = graph.filter_nodes(lambda nodes: nodes.data["unit"]==1) 
        snode_id = graph.filter_nodes(lambda nodes: nodes.data["unit"]==2) # both word node and speaker node
        # the start state
        wnode_id_id = graph.nodes[wnode_id].data['id'] 
        wnode_id_id_list = wnode_id_id.tolist()
        word_list = self.vocab.unmap(wnode_id_id_list)
        # word_list_block = []
        word_embed_list = []
        #print(word_list)
        for word in word_list:
            #print(len(self.embed_dic[word]))
            word_embed_list.append(self.embed_dic_word[word])
            word = word + " " + "[SEP]"
            #word_list_block.append(word)
        pnode_id_id = graph.nodes[pnode_id].data['id'] 
        pnode_id_id_list = pnode_id_id.tolist()
        phrase_list = self.vocab_ph.unmap(pnode_id_id_list)
        pnode_id_id = graph.nodes[pnode_id].data['id'] 
        pnode_id_id_list = pnode_id_id.tolist()
        phrase_list = self.vocab_ph.unmap(pnode_id_id_list)
        #out_phrase = self. (phrase_list)
        phrase_list_block, phrase_embed_list = [], []
        phrase_one, phrase_one_ind = [],[]#phrase_one記錄頻率爲一的短語, phrase_one_ind頻次爲一時的位置
        for ind, phrase in enumerate(phrase_list):
            #print(phrase)
            #if self.phrase_dic_dic[phrase] != 1:
            phrase_embed_list.append(self.embed_dic_phrase[phrase])
        snode_id_id = graph.nodes[snode_id].data['id'] 
        snode_id_id_list = snode_id_id.tolist()
        sent_list = self.vocab_sent.unmap(snode_id_id_list)
        sent_list_block, sent_embed_list = [], []
        for sent in sent_list:
            sent_embed_list.append(self.embed_dic_sent[sent])
            # sent = "[CLS]" + " " + sent + " " + "[SEP]"
            # sent_list_block.append(sent)
        #print(len(phrase_embed_list))
        #print(len(phrase_embed_list), len(word_embed_list), len(sent_embed_list))
        phrase_embed = torch.tensor(phrase_embed_list, dtype = torch.float32, device=config.device)
        word_embed = torch.tensor(word_embed_list, dtype = torch.float32, device=config.device)
        sent_embed = torch.tensor(sent_embed_list, dtype = torch.float32, device=config.device)

        graph.nodes[wnode_id].data["feat"] = word_embed.to(device=config.device)####word_embed#word_state
        graph.nodes[pnode_id].data["feat"] = phrase_embed.to(device=config.device)####phrase_embed#phrase_state
        graph.nodes[snode_id].data["feat"] = sent_embed.to(device=config.device)####sent_embed#sent_state
        wpedge_id = graph.filter_edges(lambda edges: (edges.src["unit"] == 0) & (edges.dst["unit"] == 1))
        pwedge_id = graph.filter_edges(lambda edges: (edges.src["unit"] == 1) & (edges.dst["unit"] == 0))
        psedge_id = graph.filter_edges(lambda edges: edges.data["dtype"] == 13)
        spedge_id = graph.filter_edges(lambda edges: (edges.src["unit"] == 2) & (edges.dst["unit"] == 1))
        wsedge_id = graph.filter_edges(lambda edges: (edges.src["unit"] == 0) & (edges.dst["unit"] == 2))
        swedge_id = graph.filter_edges(lambda edges: (edges.src["unit"] == 2) & (edges.dst["unit"] == 0))
        p2ppedge_id = graph.filter_edges(lambda edges: edges.data["dtype"] == 4)
        pp2pedge_id = graph.filter_edges(lambda edges: edges.data["dtype"] == 1)
        edgs_all = graph.edges()
        #print(wpedge_id)
        #print("wpedge_id", wpedge_id.shape)
        wnode_id_list = wnode_id.tolist()
        pnode_id_list = pnode_id.tolist()
        snode_id_list = snode_id.tolist()
        all_list = wnode_id_list + pnode_id_list + snode_id_list
        #print(len(all_list))
        edgs_all_list1 = edgs_all[0].tolist()
        edgs_all_list2 = edgs_all[1].tolist()
        edgs_all_list = edgs_all_list1 + edgs_all_list2
        edgs_all_list = list(set(edgs_all_list))
        wp_dict = {}
        wp_node_w = edgs_all[0][wpedge_id]
        wp_node_w1 = wp_node_w.tolist()
        wp_node_w = torch.tensor(list(set(wp_node_w1)), dtype = torch.int64, device=config.device)
        #print(wp_node_w)
        #print("wp_node_w", wp_node_w.shape)
        wpedge_id_list = wpedge_id.tolist()
        #print(wpedge_id_list)
        wp_node_p = edgs_all[1][wpedge_id]

        wp_node_p1 = wp_node_p.tolist()
        wp_node_p = torch.tensor(list(set(wp_node_p1)), dtype = torch.int64, device=config.device)

        pw_node_p = edgs_all[0][pwedge_id]
        pw_node_p = pw_node_p.tolist()
        pw_node_p = torch.tensor(list(set(pw_node_p)), dtype = torch.int64, device=config.device)
        pw_node_w = edgs_all[1][pwedge_id]
        pw_node_w = pw_node_w.tolist()
        pw_node_w = torch.tensor(list(set(pw_node_w)), dtype = torch.int64, device=config.device)
        
        ws_node_w = edgs_all[0][wsedge_id]
        ws_node_w = ws_node_w.tolist()
        ws_node_w = torch.tensor(list(set(ws_node_w)), dtype = torch.int64, device=config.device)
        ws_node_s = edgs_all[1][wsedge_id]
        ws_node_s = ws_node_s.tolist()
        ws_node_s = torch.tensor(list(set(ws_node_s)), dtype = torch.int64, device=config.device)
        
        sw_node_s = edgs_all[0][swedge_id]
        sw_node_s = sw_node_s.tolist()
        sw_node_s = torch.tensor(list(set(sw_node_s)), dtype = torch.int64, device=config.device)
        sw_node_w = edgs_all[1][swedge_id]
        sw_node_w = sw_node_w.tolist()
        sw_node_w = torch.tensor(list(set(sw_node_w)), dtype = torch.int64, device=config.device)
        
        
        p_node_p = edgs_all[0][p2ppedge_id]
        p_node_p1 = p_node_p.tolist()
        p2pp_node_p = torch.tensor(list(set(p_node_p1)), dtype = torch.int64, device=config.device)
        p_node_pp = edgs_all[1][p2ppedge_id]
        p_node_pp1 = p_node_pp.tolist()
        p2pp_node_pp = torch.tensor(list(set(p_node_pp1)), dtype = torch.int64, device=config.device)
        # pp_dict = {}
        # for ind_wp, wp in enumerate(p_node_pp1):
            # if pp_dict.get(wp, 0) == 0:
                # pp_dict[wp] = [p_node_p1[ind_wp]]
            # else:
                # pp_dict[wp].append(p_node_p1[ind_wp])

        # for key in pp_dict.keys():

            # wp = torch.tensor(pp_dict[key], dtype = torch.int64, device=config.device)
            # wp_emd = graph.nodes[wp].data["feat"]
            # print(wp_emd)
        
        
        
        
        ps_node_p = edgs_all[0][psedge_id]
        ps_node_p = ps_node_p.tolist()
        ps_node_p = torch.tensor(list(set(ps_node_p)), dtype = torch.int64, device=config.device)
        ps_node_s = edgs_all[1][psedge_id]
        ps_node_s = ps_node_s.tolist()
        ps_node_s = torch.tensor(list(set(ps_node_s)), dtype = torch.int64, device=config.device)
        sp_node_s = edgs_all[0][spedge_id]
        sp_node_s = sp_node_s.tolist()
        sp_node_s = torch.tensor(list(set(sp_node_s)), dtype = torch.int64, device=config.device)
        sp_node_p = edgs_all[1][spedge_id]
        sp_node_p = sp_node_p.tolist()
        sp_node_p = torch.tensor(list(set(sp_node_p)), dtype = torch.int64, device=config.device)
        #edgs_list_l = edgs[0].tolist()
        #edgs_list_r = edgs[1].tolist()
        #print(wp_node_w)
        # word_state = graph.nodes[wnode_id].data["feat"]
        # phrase_state = graph.nodes[pnode_id].data["feat"]
        # sent_state = graph.nodes[snode_id].data["feat"]
        # wp_node_w = graph.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
        # wp_node_p = graph.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
        # ws_node_w = graph.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
        # ws_node_s = graph.filter_nodes(lambda nodes: nodes.data["unit"] == 2)
        for i in range(self.config.ggcn_n_iter):
            #print("sent_state1", sent_state.shape)
            word_state = graph.nodes[wp_node_w].data["feat"]
            phrase_state = graph.nodes[wp_node_p].data["feat"]
            phrase_state = self.word2phrase(graph, word_state, phrase_state, word_state, wp_node_w, wp_node_p, wpedge_id)
            graph.nodes[wp_node_p].data["feat"] = phrase_state
            
            
            
            
            # P_state = graph.nodes[p2pp_node_p].data["feat"]
            # PP_state = graph.nodes[p2pp_node_pp].data["feat"]
            # phrase_state = self.phrase2phrasephrase(graph, P_state, PP_state, PP_state, p2pp_node_p, p2pp_node_pp, p2ppedge_id)
            # graph.nodes[p2pp_node_pp].data["feat"] = phrase_state
            
            
            
            
            # word_state = graph.nodes[pw_node_w].data["feat"]
            # phrase_state = graph.nodes[pw_node_p].data["feat"]
            # word_state = self.phrase2word(graph, phrase_state, word_state, word_state, pw_node_p, pw_node_w, pwedge_id)
            # graph.nodes[pw_node_w].data["feat"] = word_state

            ####print("111111111111111111111111111111111111111111111111111111111111111111111111111")
            phrase_state = graph.nodes[ps_node_p].data["feat"]
            sent_state = graph.nodes[ps_node_s].data["feat"]
            sent_state = self.phrase2sent(graph, phrase_state, sent_state, word_state, ps_node_p, ps_node_s, psedge_id)
            graph.nodes[ps_node_s].data["feat"] = sent_state
            
            
            
            
            word_state = graph.nodes[ws_node_w].data["feat"]
            sent_state = graph.nodes[ws_node_s].data["feat"]
            sent_state = self.word2sent(graph, word_state, sent_state, word_state, ws_node_w, ws_node_s, wsedge_id)
            graph.nodes[ws_node_s].data["feat"] = sent_state
            
            
            

            
            






        word_state = graph.nodes[wnode_id].data["feat"]
        phrase_state = graph.nodes[pnode_id].data["feat"]
        sent_state = graph.nodes[snode_id].data["feat"]
        return None

    def set_wordNode_feature(self, graph, local_feature_w):   #, graph1
        wnode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"]==0) # only word node
        #wnode_id1 = graph1.filter_nodes(lambda nodes: nodes.data["dtype"]==0)
        wid = graph.nodes[wnode_id].data["id"]  # [n_wnodes]
        w_embed = self.word_embedding(wid)  # [n_wnodes, D]
        #w_embed_1 = torch.unsqueeze(w_embed, 1)
        w_embed_2 = torch.cat([w_embed, w_embed], 1)


        #print("local_feature_w", local_feature_w.shape)
        graph.nodes[wnode_id].data["embed"] = local_feature_w          #对wnode_id赋值embeding
        #graph1.nodes[wnode_id1].data["embed"] = local_feature_w
        return w_embed

    
    def set_wordSentEdge_feature(self, graph):
        #Intialize word sent edge
        wsedge_id = graph.filter_edges(lambda edges: edges.data["dtype"] == 2)   # both word node and speaker node
        ws_edge = graph.edges[wsedge_id].data['ws_link']
        graph.edges[wsedge_id].data["ws_embed"] = self.ws_embed(ws_edge)
    def set_SentphraseEdge_feature(self, graph):
        #Intialize word sent edge
        spedge_id = graph.filter_edges(lambda edges: edges.data["dtype"] == 3)   # both word node and speaker node
        sp_edge = graph.edges[spedge_id].data['ws_link']
        graph.edges[spedge_id].data["sp_embed"] = self.ws_embed(sp_edge)
    def set_wordPhraseEdge_feature(self, graph):
        #Intialize word sent edge
        wpedge_id = graph.filter_edges(lambda edges: edges.data["dtype"] == 0)   # both word node and speaker node
        wp_edge = graph.edges[wpedge_id].data['ws_link']
        graph.edges[wpedge_id].data["wp_embed"] = self.ws_embed(wp_edge)
    def set_phrasePhraseEdge_feature(self, graph):
        #Intialize word sent edge
        ppedge_id = graph.filter_edges(lambda edges: edges.data["dtype"] == 1)   # both word node and speaker node
        pp_edge = graph.edges[ppedge_id].data['ws_link']
        #print("huahua", graph.edges[ppedge_id].data["pp_embed"])
        graph.edges[ppedge_id].data["pp_embed"] = self.ws_embed(pp_edge)
    def set_PhraseNode_feature(self, graph, batch, local_feature_p): #, graph1

        phrase_feature = local_feature_p.reshape(-1, local_feature_p.size(-1))
        phrase_feature = phrase_feature[batch['utter_p_index']] # (batch * total_number_utt, glstm_hidden_dim*2)     
        pnode_id = graph.filter_nodes(lambda nodes: nodes.data['dtype'] == 1) # only sent nodes
        #pnode_id1 = graph1.filter_nodes(lambda nodes: nodes.data['dtype'] == 1)
        #print("self.sent_feature_proj(phrase_feature).shape", phrase_feature.shape, self.sent_feature_proj(phrase_feature).shape)
        #graph1.nodes[pnode_id1].data['init_state_p'] = phrase_feature
        graph.nodes[pnode_id].data['init_state_p'] = phrase_feature#self.sent_feature_proj(phrase_feature)#self.sent_feature_proj(phrase_feature)#phrase_feature#self.sent_feature_proj(phrase_feature)

    def set_sentNode_feature(self, graph, batch, local_feature_s):   #, graph1
        #print("123321local_feature", local_feature.shape)
        #sent_feature, _ = self.glstm(local_feature) # (batch, max_number_utt, glstm_hidden_dim*2)
        #print("sent_feature1", sent_feature.shape)
        #sent_feature = sent_feature * batch['conv_mask'][:,:,0].unsqueeze(-1) # masking 
        sent_feature = local_feature_s.reshape(-1, local_feature_s.size(-1))
        #print("sent_feature1", sent_feature.shape)
        sent_feature = sent_feature[batch['utter_index']] # (batch * total_number_utt, glstm_hidden_dim*2)       
        #print("sent_feature2", sent_feature.shape)
        snode_id = graph.filter_nodes(lambda nodes: nodes.data['dtype'] == 2) # only sent node
        ###snode_id1 = graph1.filter_nodes(lambda nodes: nodes.data['dtype'] == 2)
        #print("snode_id", snode_id)
        #print("sent_feature", self.sent_feature_proj(sent_feature).shape, sent_feature.shape)
        #print("init_state_s", sent_feature.shape)
        graph.nodes[snode_id].data['init_state_s'] = sent_feature#self.sent_feature_proj(sent_feature)#sent_feature#self.sent_feature_proj(sent_feature) self.sent_feature_proj(sent_feature)#
        ####graph1.nodes[snode_id1].data['init_state_s'] = sent_feature



