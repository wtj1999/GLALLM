import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.tools import reparameterize


class ParameterGenerator(nn.Module):
    def __init__(self, configs, input_dim):
        super(ParameterGenerator, self).__init__()
        self.num_nodes = configs.num_nodes
        self.memory_size = configs.memory_size
        self.fused_dim = configs.fused_embed_dim
        self.input_dim = input_dim

        self.weight_generator = nn.Sequential(*[
            nn.Linear(self.memory_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, self.input_dim * self.fused_dim)
        ])

        self.bias_generator = nn.Sequential(*[
            nn.Linear(self.memory_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, self.input_dim * self.fused_dim)
        ])

    def forward(self, z):
        weights = self.weight_generator(z).reshape([-1, self.num_nodes, self.input_dim, self.fused_dim])
        biases = self.bias_generator(z).reshape([-1, self.num_nodes, self.input_dim, self.fused_dim])

        return weights, biases

class LinearCustom(nn.Module):
    def __init__(self):
        super(LinearCustom, self).__init__()

    def forward(self, input, parameters): #[B, N, T, D], [B, N, D_in, D_out]
        weights, biases = parameters[0], parameters[1]

        return input @ weights

class STA_Fusion(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.num_nodes = configs.num_nodes
        self.memory_size = configs.memory_size
        self.global_embed_dim = configs.global_embed_dim
        self.local_embed_dim = configs.local_embed_dim
        self.fused_dim = configs.fused_embed_dim
        self.mu_q = nn.Parameter(torch.randn(self.num_nodes, self.memory_size), requires_grad=True)
        self.logvar_q = nn.Parameter(torch.randn(self.num_nodes, self.memory_size), requires_grad=True)

        self.mu_k = nn.Parameter(torch.randn(self.num_nodes, self.memory_size), requires_grad=True)
        self.logvar_k = nn.Parameter(torch.randn(self.num_nodes, self.memory_size), requires_grad=True)

        self.mu_v = nn.Parameter(torch.randn(self.num_nodes, self.memory_size), requires_grad=True)
        self.logvar_v = nn.Parameter(torch.randn(self.num_nodes, self.memory_size), requires_grad=True)

        self.q_parameter_generator = ParameterGenerator(configs, self.global_embed_dim)
        self.k_parameter_generator = ParameterGenerator(configs, self.local_embed_dim)
        self.v_parameter_generator = ParameterGenerator(configs, self.local_embed_dim)

        self.proj = nn.Linear(self.global_embed_dim, self.fused_dim)
        self.linear = LinearCustom()
        self.norm = nn.LayerNorm(self.fused_dim)

    def forward(self, global_embed, local_embed):
        q_sample = reparameterize(self.mu_q, self.logvar_q)
        k_sample = reparameterize(self.mu_k, self.logvar_k)
        v_sample = reparameterize(self.mu_v, self.logvar_v)

        q_parameter = self.q_parameter_generator(q_sample)
        k_parameter = self.k_parameter_generator(k_sample)
        v_parameter = self.v_parameter_generator(v_sample)

        Q = self.linear(global_embed, q_parameter)
        K = self.linear(local_embed, k_parameter)
        V = self.linear(local_embed, v_parameter)

        Att = F.softmax(Q @ K.transpose(2, 3) / torch.sqrt(torch.tensor(K.size(-1))), dim=-1)
        out = Att @ V

        out = out + self.proj(global_embed)
        out = self.norm(out)

        return out


