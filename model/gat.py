# Some of the classes are adapted from brxx12/HeterSumGraph

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import config

class WSWGAT(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, attn_drop_out, ffn_inner_hidden_size, ffn_drop_out, feat_embed_size, layerType):
        super().__init__()
        self.layerType = layerType
        if layerType == "W2S":
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out, feat_embed_size, layer=WSGATLayer)
        elif layerType == "S2W":
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out, feat_embed_size, layer=SWGATLayer)
        elif layerType == "P2W":
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out, feat_embed_size, layer=PWGATLayer)
        elif layerType == "W2P":
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out, feat_embed_size, layer=WPGATLayer)
        elif layerType == "P2S":
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out, feat_embed_size, layer=PSGATLayer)
        elif layerType == "S2P":
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out, feat_embed_size, layer=SPGATLayer)
        #elif layerType == "S2P":
            #self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out, feat_embed_size, layer=SPGATLayer)
        elif layerType == "PP2P":
            self.layer = MultiHeadSGATLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out)#句法树上面的短语往下传递
        elif layerType == "P2PP":
            self.layer = MultiHeadPGATLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out)    #下面短语往上走
        elif layerType == "P2A":
            self.layer = MultiHeadPAATLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out)
        elif layerType == "N2P":
            self.layer = MultiHeadNPATLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out)
        else:
            raise NotImplementedError("GAT Layer has not been implemented!")

        self.ffn = PositionwiseFeedForward(out_dim, ffn_inner_hidden_size, ffn_drop_out)

    def forward(self, g, w, s, wb, w_node, s_node, edge_id):
        if self.layerType == "W2S":
            origin, neighbor, spare  = s, w, wb
            h = F.elu(self.layer(g, neighbor, origin, spare, w_node, s_node, edge_id))
            h= h + origin
        elif self.layerType == "S2W":
            origin, neighbor, spare = s, w, wb
            h = F.elu(self.layer(g, neighbor, origin, spare, w_node, s_node, edge_id))
            h= h + origin
        elif self.layerType == "P2W":
            origin, neighbor, spare = s, w, wb
            h = F.elu(self.layer(g, neighbor, origin, spare, w_node, s_node, edge_id))
            h= h + origin
        elif self.layerType == "W2P":
            origin, neighbor, spare = s, w, wb
            h = F.elu(self.layer(g, neighbor, origin, spare, w_node, s_node, edge_id))
            h= h + origin
        elif self.layerType == "P2S":
            origin, neighbor, spare = s, w, wb
            h = F.elu(self.layer(g, neighbor, origin, spare, w_node, s_node, edge_id))
            h= h + origin
        elif self.layerType == "S2P":
            origin, neighbor, spare = s, w, wb
            h = F.elu(self.layer(g, neighbor, origin, spare, w_node, s_node, edge_id))
            h= h + origin
        elif self.layerType == "P2PP":
            #assert torch.equal(w, s)
            origin, origin1, neighbor = s, w, w
            h = F.elu(self.layer(g, neighbor, origin1, w_node, s_node, edge_id))
            h= h + origin
        elif self.layerType == "PP2P":
            assert torch.equal(w, s)
            origin, neighbor = w, s
            h = F.elu(self.layer(g, neighbor, origin, w_node, s_node, edge_id))
            #h= h + origin
        elif self.layerType == "P2A":
            assert torch.equal(w, s)
            origin, neighbor = w, s
            h = F.elu(self.layer(g, neighbor, origin, w_node, s_node, edge_id))
            #h= h + origin
        elif self.layerType == "N2P":
            assert torch.equal(w, s)
            origin, neighbor = w, s
            h = F.elu(self.layer(g, neighbor, origin))
            #h= h + origin
        else:
            origin, neighbor = None, None

        #h = F.elu(self.layer(g, neighbor, origin))
        #h= h + origin
        h = self.ffn(h.unsqueeze(0)).squeeze(0)
        return h

class MultiHeadPAATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, attn_drop_out, merge='cat'):   ###merge='cat'
        super(MultiHeadPAATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(PAATLayer(in_dim, out_dim))  # [n_nodes, hidden_size]
        self.merge = merge
        self.dropout = nn.Dropout(attn_drop_out)

    def forward(self, g, h, s):
        head_outs = [attn_head(g, self.dropout(h), self.dropout(s)) for attn_head in self.heads]  # n_head * [n_nodes, hidden_size]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)  # [n_nodes, hidden_size * n_head]
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))
class MultiHeadNPATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, attn_drop_out, merge='cat'):   ###merge='cat'
        super(MultiHeadNPATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(NPATLayer(in_dim, out_dim))  # [n_nodes, hidden_size]
        self.merge = merge
        self.dropout = nn.Dropout(attn_drop_out)

    def forward(self, g, h, s):
        head_outs = [attn_head(g, self.dropout(h)) for attn_head in self.heads]  # n_head * [n_nodes, hidden_size]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)  # [n_nodes, hidden_size * n_head]
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))
class MultiHeadPGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, attn_drop_out, merge='cat'):   ###merge='cat'
        super(MultiHeadPGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(PGATLayer(in_dim, out_dim))  # [n_nodes, hidden_size]
        self.merge = merge
        self.dropout = nn.Dropout(attn_drop_out)

    def forward(self, g, h, s, w_node, s_node, edge_id):
        head_outs = [attn_head(g, self.dropout(h), self.dropout(s), w_node, s_node, edge_id) for attn_head in self.heads]  # n_head * [n_nodes, hidden_size]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)  # [n_nodes, hidden_size * n_head]
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))


class MultiHeadSGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, attn_drop_out, merge='cat'):   ###merge='cat'
        super(MultiHeadSGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(
                SGATLayer(in_dim, out_dim))  # [n_nodes, hidden_size]
        self.merge = merge
        self.dropout = nn.Dropout(attn_drop_out)

    def forward(self, g, h, s, w_node, s_node, edge_id):
        head_outs = [attn_head(g, self.dropout(h), w_node, s_node, edge_id) for attn_head in self.heads]  # n_head * [n_nodes, hidden_size]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)  # [n_nodes, hidden_size * n_head]
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))


class MultiHeadLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, attn_drop_out, feat_embed_size, layer, merge='cat'):#merge='cat'
        super(MultiHeadLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(layer(in_dim, out_dim, feat_embed_size))  # [n_nodes, hidden_size]
        self.merge = merge
        self.dropout = nn.Dropout(attn_drop_out)

    def forward(self, g, h, s, wb, w_node, s_node, edge_id):
        head_outs = [attn_head(g, self.dropout(h), self.dropout(s), self.dropout(wb), w_node, s_node, edge_id) for attn_head in self.heads]  # n_head * [n_nodes, hidden_size]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            result = torch.cat(head_outs, dim=1)  # [n_nodes, hidden_size * n_head]
        else:
            # merge using average
            result = torch.mean(torch.stack(head_outs))
        return result


class SGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, weight=0):
        super(SGATLayer, self).__init__()
        self.weight = weight
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)  # [edge_num, 2 * out_dim]
        wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]
        return {'e': wa}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        #print("nodes", nodes.mailbox['e'])
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    def forward(self, g, h, ppnode_id, pnode_id, pedge_id):
        snode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
        sedge_id = g.filter_edges(lambda edges: edges.data["dtype"] == 1)
        z = self.fc(h)
        g.nodes[ppnode_id].data['z'] = z
        g.apply_edges(self.edge_attention, edges=sedge_id)
        g.pull(pnode_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        h = g.ndata.pop('sh')
        return h[pnode_id]
class NPATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, weight=0):
        super(NPATLayer, self).__init__()
        self.weight = weight
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)  # [edge_num, 2 * out_dim]
        wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]
        return {'e': wa}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    def forward(self, g, h, s):
        snode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
        sedge_id = g.filter_edges(lambda edges: edges.data["dtype"] == 6)
        z = self.fc(h)
        g.nodes[snode_id].data['z'] = z
        g.apply_edges(self.edge_attention, edges=sedge_id)
        g.pull(snode_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        h = g.ndata.pop('sh')
        return h[snode_id]
class PAATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, weight=0):
        super(PAATLayer, self).__init__()
        self.weight = weight
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)  # [edge_num, 2 * out_dim]
        wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]
        return {'e': wa}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    def forward(self, g, h, s):
        snode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
        sedge_id = g.filter_edges(lambda edges: edges.data["dtype"] == 5)
        z = self.fc(h)
        g.nodes[snode_id].data['z'] = z
        g.apply_edges(self.edge_attention, edges=sedge_id)
        g.pull(snode_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        h = g.ndata.pop('sh')
        return h[snode_id]
class PGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, weight=0):
        super(PGATLayer, self).__init__()
        self.weight = weight
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)  # [edge_num, 2 * out_dim]
        
        wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]
        wa_list = wa.tolist()
        #print("alpha", wa_list)
        return {'e': wa}

    def message_func(self, edges):
        #print("1111", edges.src['z'].shape)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        mailbox_e = []
        node_e = nodes.mailbox['e'].tolist()
        for ind_list, weig_list in enumerate(node_e):
            mailbox_e.append([])
            for ind_e, weig in enumerate(weig_list):
                if weig[0] == 0:
                    mailbox_e[-1].append([-1000.0])
                else:
                    mailbox_e[-1].append(weig)
        mailbox_e = torch.tensor(mailbox_e, device=config.device)
        alpha = F.softmax(mailbox_e, dim=1)
        # alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        ###e_list = nodes.mailbox['e'].tolist()
        ####alpha = F.softmax(nodes.mailbox['e'], dim=1)
        ###alpha_list = alpha.tolist()
        #print("alpha", nodes.mailbox['e'])#nodes.mailbox['e'])nodes.mailbox['e']
        #print("alpha2", nodes.mailbox['z'].shape)#nodes.mailbox['e']
        # nodes_sum = torch.sum(nodes.mailbox['z'], dim=2)
        # nodes_sum_list = nodes_sum.tolist()
        #print(nodes_sum_list)
        #z_list = nodes.mailbox['z'].tolist()
        ####for ind, i in enumerate(e_list):
            ####print("111", i)
            #print(alpha_list[ind])
            ####print("222", nodes_sum_list[ind])
        ###h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        #print("alpha1", h.shape)
        return {'sh': h}

    def forward(self, g, h, s, pnode_id, ppnode_id, sedge_id): #s是word，h是p

        z = self.fc(h)
        #z1 = self.fc(s)
        #z2 = torch.cat([z, z1], dim=0)
        
        #spnode_id = torch.cat([snode_id, wnode_id], dim=0)
        #print("111", z2.shape)
        ####print("2222", spnode_id.shape)
        #print("1", snode_id.shape)
        #print("2", wnode_id.shape)
        g.nodes[pnode_id].data['z'] = z
        #g.nodes[wnode_id].data['z'] = z1
        
        g.apply_edges(self.edge_attention, edges=sedge_id)
        g.pull(ppnode_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        h = g.ndata.pop('sh')
        return h[ppnode_id]


class WSGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, feat_embed_size):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)

        self.feat_fc = nn.Linear(feat_embed_size, out_dim, bias=False)
        ####self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        ###self.ne_fc = nn.Linear(3 * out_dim, out_dim, bias=False)

    def edge_attention(self, edges):
        ####print("1233321123", edges.data["ws_embed"])
        dfeat = self.feat_fc(edges.data["ws_embed"])                  # [edge_num, out_dim]
        ####print("123321123321", edges.dst['z'],size())  ####,edges.src['z']   , dfeat
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)  # , dfeat[edge_num, 3 * out_dim], edges.dst['z'], dfeat
        wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]
        ####print("1233321123", wa.size())
        return {'e': wa}

    def message_func(self, edges):
        #print("edge e ", edges.src['z'].size())#.size()
        #print(edges.src['z'].size)
        ####dfeat = self.feat_fc(edges.data["ws_embed"]) 
        ####z3 = torch.cat([edges.src['z'], dfeat], dim=1)
        ####z4 = self.ne_fc(z3)
        #print("edge e 2", z4.size(), z3.size())
        ####return {'z': z4, 'e': edges.data['e']}
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):

        mailbox_e = []
        node_e = nodes.mailbox['e'].tolist()
        for ind_list, weig_list in enumerate(node_e):
            mailbox_e.append([])
            for ind_e, weig in enumerate(weig_list):
                if weig[0] == 0:
                    mailbox_e[-1].append([-1000.0])
                else:
                    mailbox_e[-1].append(weig)
        mailbox_e = torch.tensor(mailbox_e, device=config.device)
        alpha = F.softmax(mailbox_e, dim=1)
        # alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    def forward(self, g, h, s, wb, wnode_id, snode_id, wsedge_id):
        # wnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
        # snode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 2)
        # wsedge_id = g.filter_edges(lambda edges: (edges.src["unit"] == 0) & (edges.dst["unit"] == 2))
        # edgs = g.edges()
        # edgs_list_l = edgs[0][wsedge_id]
        # wnode_id = torch.tensor(list(set(edgs_list_l)), dtype = torch.int64, device=config.device)
        # edgs_list_r = edgs[1][wsedge_id]
        # snode_id = torch.tensor(list(set(edgs_list_r)), dtype = torch.int64, device=config.device)
        #print(len(wsedge_id))
        #print(wsedge_id)
        # print("id in WSGATLayer")
        # print(wnode_id, snode_id, wsedge_id)
        ####print("123321123321", h.size())
        z = self.fc(h)
        #y = self.fc(s)
        #print("123321123321", wsedge_id.size())
        #print("2222123321123321", z.size())
        g.nodes[wnode_id].data['z'] = z
        #g.nodes[snode_id].data['z'] = y
        g.apply_edges(self.edge_attention, edges=wsedge_id)
        g.pull(snode_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        h = g.ndata.pop('sh')
        return h[snode_id]
class SPGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, feat_embed_size):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)

        self.feat_fc = nn.Linear(feat_embed_size, out_dim)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        dfeat = self.feat_fc(edges.data["sp_embed"])  # [edge_num, out_dim]
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)  #, dfeat [edge_num, 3 * out_dim]
        wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]
        return {'e': wa}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    def forward(self, g, h, s, wb, snode_id, pnode_id,psedge_id):
        snode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 2)
        pnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
        psedge_id = g.filter_edges(lambda edges: (edges.src["unit"] == 2) & (edges.dst["unit"] == 1))#src为源，dst为目的
        #print(h.shape)
        z = self.fc(h)
        #print(z.shape)
        #print(len(snode_id), len(snode_id), z.shape)
        g.nodes[snode_id].data['z'] = z
        g.apply_edges(self.edge_attention, edges=psedge_id)
        g.pull(pnode_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        h = g.ndata.pop('sh')
        return h[pnode_id]
class PSGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, feat_embed_size):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)

        self.feat_fc = nn.Linear(feat_embed_size, out_dim)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        dfeat = self.feat_fc(edges.data["sp_embed"])  # [edge_num, out_dim]
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)  # [edge_num, 3 * out_dim], dfeat
        wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]
        return {'e': wa}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        mailbox_e = []
        node_e = nodes.mailbox['e'].tolist()
        for ind_list, weig_list in enumerate(node_e):
            mailbox_e.append([])
            for ind_e, weig in enumerate(weig_list):
                if weig[0] == 0:
                    mailbox_e[-1].append([-1000.0])
                else:
                    mailbox_e[-1].append(weig)
        mailbox_e = torch.tensor(mailbox_e, device=config.device)
        alpha = F.softmax(mailbox_e, dim=1)
        # alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    def forward(self, g, h, s, wb, pnode_id, snode_id, psedge_id):
        #print(pnode_id.shape)
        #print("123321", psedge_id.shape)
        z = self.fc(h)
        g.nodes[pnode_id].data['z'] = z
        g.apply_edges(self.edge_attention, edges=psedge_id)
        g.pull(snode_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        h = g.ndata.pop('sh')
        return h[snode_id]
class PWGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, feat_embed_size):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)

        self.feat_fc = nn.Linear(feat_embed_size, out_dim)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        dfeat = self.feat_fc(edges.data["wp_embed"])  # [edge_num, out_dim]
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)  # [edge_num, 3 * out_dim], dfeat
        wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]
        return {'e': wa}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        mailbox_e = []
        node_e = nodes.mailbox['e'].tolist()
        for ind_list, weig_list in enumerate(node_e):
            mailbox_e.append([])
            for ind_e, weig in enumerate(weig_list):
                if weig[0] == 0:
                    mailbox_e[-1].append([-1000.0])
                else:
                    mailbox_e[-1].append(weig)
        mailbox_e = torch.tensor(mailbox_e, device=config.device)
        alpha = F.softmax(mailbox_e, dim=1)
        # alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    def forward(self, g, h, s, wb, pnode_id, wnode_id, pwedge_id):
        #wnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
        #pnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
        # pwedge_id = g.filter_edges(lambda edges: (edges.src["unit"] == 1) & (edges.dst["unit"] == 0))   #src源节点，dst目标节点
        # edgs = g.edges()
        # edgs_list_l = edgs[0][pwedge_id]
        # wnode_id = torch.tensor(list(set(edgs_list_l)), dtype = torch.int64, device=config.device)
        # edgs_list_r = edgs[1][pwedge_id]
        # pnode_id = torch.tensor(list(set(edgs_list_r)), dtype = torch.int64, device=config.device)
        z = self.fc(h)
        g.nodes[pnode_id].data['z'] = z
        g.apply_edges(self.edge_attention, edges=pwedge_id)
        g.pull(wnode_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        h = g.ndata.pop('sh')
        return h[wnode_id]
class WPGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, feat_embed_size):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)

        self.feat_fc = nn.Linear(feat_embed_size, out_dim)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        
    def edge_attention(self, edges):
        dfeat = self.feat_fc(edges.data["wp_embed"])  # [edge_num, out_dim]
        
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)  # [edge_num, 3 * out_dim], dfeat
        #print("12331", edges.src['z'].shape)
        wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]
        #print(wa)
        return {'e': wa}

    def message_func(self, edges):
        edges_src = torch.sum(edges.src['z'], dim=1)
        edges_src = edges_src.tolist()
        a = []
        for ind, i in enumerate(edges_src):
            if i == 0:
                a.append(i)
            else:
                a.append(1.0)
        #print(a)
        #print("321123", edges.src['z'].shape)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        #alpha = F.softmax(nodes.mailbox['e'], dim=1)
        mailbox_e = []
        node_e = nodes.mailbox['e'].tolist()
        node_z = nodes.mailbox['z'].tolist()
        for ind_list, weig_list in enumerate(node_e):
            mailbox_e.append([])
            for ind_e, weig in enumerate(weig_list):
                if weig[0] == 0:
                    mailbox_e[-1].append([-1000.0])

                else:
                    mailbox_e[-1].append(weig)

        mailbox_e = torch.tensor(mailbox_e, device=config.device)
        alpha = F.softmax(mailbox_e, dim=1)#mailbox_e
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        #h = torch.sum(alpha * mailbox_z, dim=1)
        #print(alpha)
        return {'sh': h}

    def forward(self, g, h, s, wb, wnode_id, pnode_id, pwedge_id):

        edgs = g.edges()
        node_all = g.nodes()

        #print("11111111", h.shape)
        z = self.fc(h)
        
        #print("pwedge_id", pwedge_id.shape)
        g.nodes[wnode_id].data['z'] = z
        g.apply_edges(self.edge_attention, edges=pwedge_id)
        #pwedge_id_w= g.ndata['e']
        #print("pwedge_id_w", pwedge_id_w.shape)
        g.pull(pnode_id, self.message_func, self.reduce_func)

        #print("s_z", s_z.shape)
        #s_z = torch.sum(s_z, dim=1)
        #s_z_list = s_z.tolist()
        egde_id, egde_id_no0 = [], []
        #print("s_z_list", len(s_z_list))
        # for ind, s in enumerate(s_z_list):
            # if s == 0:
                #print("111111111111")
                # egde_id.append(ind)
            # else:
                # egde_id_no0.append(ind)
        ####print("egde_id", len(egde_id))
        ####print("egde_id_no0", len(egde_id_no0))
        # egde_id_id = torch.tensor(egde_id, dtype = torch.int64, device=config.device)
        # egde_id_no = torch.tensor(egde_id_no0, dtype = torch.int64, device=config.device)
        # pwedge_id_0 = pwedge_id[egde_id_id]
        # pwedge_id_no = pwedge_id[egde_id_no]
        # edgs0 = edgs[0][pwedge_id_0]
        # edgs1 = edgs[1][pwedge_id_0]
        # edgs0_no = edgs[0][pwedge_id_no]
        # edgs1_no = edgs[1][pwedge_id_no]
        # edgs1 = list(set(edgs1.tolist()))
        # edgs1_no = list(set(edgs1_no.tolist()))
        # edgs_list1 = edgs[0][pwedge_id].tolist()
        # edgs_list2 = edgs[1][pwedge_id].tolist()
        #print("edgs0", edgs0)
        #print("edgs1", edgs_list1)
        #print("edgs2", edgs_list2)
        #print("edgs2", edgs0_no)
        #print("edgs3", edgs1_no)
        #print(s_z)
        g.ndata.pop('z')
        h = g.ndata.pop('sh')
        #print("node_all", node_all.shape)
        #print("h", h.shape)
        return h[pnode_id]



class SWGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, feat_embed_size):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)

        self.feat_fc = nn.Linear(feat_embed_size, out_dim)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        dfeat = self.feat_fc(edges.data["ws_embed"])  # [edge_num, out_dim]
        ####print(edges.dst['z'])
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)  # [edge_num, 3 * out_dim], edges.dst['z'],  dfeat,  dfeat

        wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]
        ####print("123321123321", wa)
        return {'e': wa}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        mailbox_e = []
        node_e = nodes.mailbox['e'].tolist()
        node_z = nodes.mailbox['z'].tolist()
        for ind_list, weig_list in enumerate(node_e):
            mailbox_e.append([])
            for ind_e, weig in enumerate(weig_list):
                if weig[0] == 0:
                    mailbox_e[-1].append([-1000.0])

                else:
                    mailbox_e[-1].append(weig)

        mailbox_e = torch.tensor(mailbox_e, device=config.device)
        alpha = F.softmax(mailbox_e, dim=1)#mailbox_e
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        # alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    def forward(self, g, h, s, wb, snode_id, wnode_id, swedge_id):
        wnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
        snode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 2)
        swedge_id = g.filter_edges(lambda edges: (edges.src["unit"] == 2) & (edges.dst["unit"] == 0))
        z = self.fc(h)
        #y = self.fc(s)
        #g.nodes[wnode_id].data['z'] = y
        g.nodes[snode_id].data['z'] = z
        g.apply_edges(self.edge_attention, edges=swedge_id)
        g.pull(wnode_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        h = g.ndata.pop('sh')
        return h[wnode_id]
class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        assert not torch.any(torch.isnan(x)), "FFN input"
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        assert not torch.any(torch.isnan(output)), "FFN output"
        return output

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.5):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        assert not torch.any(torch.isnan(x)), "FFN input"
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        assert not torch.any(torch.isnan(output)), "FFN output"
        return output


