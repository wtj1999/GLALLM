from distutils.command.config import config

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class HyperGraphConstructor(nn.Module):
    """超图构建模块，生成数据驱动和地理驱动的关联矩阵"""

    def __init__(self, config):
        super().__init__()
        # 数据驱动的超边嵌入初始化
        self.num_nodes = config.num_nodes
        self.num_edges = config.num_edges
        self.embed_dim = config.local_embed_dim
        self.hyperedge_emb = nn.Parameter(torch.randn(self.num_edges, self.embed_dim))
        # 地理驱动的超边矩阵
        self.G_l = nn.Parameter(torch.randn(self.num_nodes, self.num_edges))
        # 线性变换参数
        self.W_Q = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.W_K = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.linear_node_embeddings = nn.Linear(config.patch_size, self.embed_dim)

    def forward(self, P_last, A):
        E = self.linear_node_embeddings(P_last)
        Q = self.W_Q(E)
        K = self.W_K(self.hyperedge_emb)
        H_data = F.softmax(Q @ K.T / torch.sqrt(torch.tensor(E.size(-1))), dim=-1)

        H_geo = A @ self.G_l
        return H_data, H_geo


class HyperGraphConv(nn.Module):
    """超图卷积层"""

    def __init__(self, config):
        super().__init__()
        self.gdep = config.graph_layer_num
        self.embed_dim = config.local_embed_dim
        self.conv_W = nn.ModuleList([nn.Linear(self.embed_dim, self.embed_dim) for i in range(self.gdep)])
        self.conv_W_g = nn.ModuleList([nn.Linear(self.embed_dim, self.embed_dim) for i in range(self.gdep)])
        self.W = nn.Linear(2 * self.embed_dim, self.embed_dim, bias=False)
        self.W_g = nn.Linear(2 * self.embed_dim, self.embed_dim, bias=False)
        self.out_linear = nn.Linear(self.gdep * self.embed_dim, self.embed_dim)

    def forward(self, X, H_data, H_geo):
        D = torch.sum(H_data, dim=-1, keepdim=True)
        D_inv = torch.where(D > 0, 1.0 / (D + 1e-8), torch.zeros_like(D))

        B = torch.sum(H_data, dim=1, keepdim=True)
        B_inv = torch.where(B > 0, 1.0 / (B + 1e-8), torch.zeros_like(B))

        A_norm = (D_inv * H_data) @ (B_inv * H_data).transpose(1, 2)

        D_geo = torch.sum(H_geo, dim=-1, keepdim=True)
        D_geo_inv = torch.where(D_geo > 0, 1.0 / (D_geo + 1e-8), torch.zeros_like(D_geo))

        B_geo = torch.sum(H_geo, dim=0, keepdim=True)
        B_geo_inv = torch.where(B_geo > 0, 1.0 / (B_geo + 1e-8), torch.zeros_like(B_geo))

        A_geo_norm = (D_geo_inv * H_geo) @ (B_geo_inv * H_geo).transpose(0, 1)


        out = []
        x_data = X
        x_geo = X

        for i in range(self.gdep):
            x_data = torch.einsum('btnd,bmn->btmd', (x_data, A_norm)) @ self.conv_W[i].weight
            x_geo = torch.einsum('btnd,mn->btmd', (x_geo, A_geo_norm)) @ self.conv_W_g[i].weight

            V = torch.sigmoid(self.W(torch.cat([x_data, X], dim=-1)) + self.W_g(torch.cat([x_geo, X], dim=-1)))
            x_fused = V * x_data + (1 - V) * x_geo
            out.append(x_fused)

        out = torch.cat(out, dim=-1)
        out = self.out_linear(out)
        return out

class STHGNN(nn.Module):
    """时空超图神经网络（Local Branch）"""
    def __init__(self, config):
        super().__init__()
        self.tem_layer_num = config.tem_layer_num
        self.embed_dim = config.local_embed_dim
        self.num_nodes = config.num_nodes
        self.hypergraph = HyperGraphConstructor(config)
        self.hyper_convs = HyperGraphConv(config)
        self.start_conv = nn.Linear(1, self.embed_dim)
        self.end_conv = nn.Linear(self.embed_dim, self.embed_dim)
        self.A = torch.tensor(np.load(os.path.join(config.root_path, config.adj_data_path)), dtype=torch.float32)
        self.norm = nn.LayerNorm(self.embed_dim)
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        new_dilation = 1
        for _ in range(self.tem_layer_num):

            # dilated convolutions
            self.filter_convs.append(nn.Conv2d(in_channels=self.embed_dim,
                                               out_channels=self.embed_dim,
                                               kernel_size=(1, 1),
                                               padding='same',
                                               dilation=(1, new_dilation)))

            self.gate_convs.append(nn.Conv2d(in_channels=self.embed_dim,
                                               out_channels=self.embed_dim,
                                               kernel_size=(1, 1),
                                               padding='same',
                                               dilation=(1, new_dilation)))

            self.skip_convs.append(nn.Linear(self.embed_dim, self.embed_dim))

            new_dilation *= 2

    def forward(self, X):
        H_data, H_geo = self.hypergraph(X.transpose(1, 2), self.A.to(X.device))
        x = self.start_conv(X.unsqueeze(-1))  
        skip = x

        for i in range(self.tem_layer_num):
            # dilated convolution
            res = x
            filter = self.filter_convs[i](res.transpose(1, 3)).transpose(1, 3)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](res.transpose(1, 3)).transpose(1, 3)
            gate = torch.sigmoid(gate)
            x = filter * gate
            skip = skip + self.skip_convs[i](x)
            x = self.hyper_convs(x, H_data, H_geo)
            x = self.norm(x)

        out = F.relu(skip)
        out = self.end_conv(out)

        return out




























