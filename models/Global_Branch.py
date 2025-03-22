import numpy as np
import torch
import torch.nn as nn
from torch import optim
from einops import rearrange

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config


class LLM4TS(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.patch_size = configs.patch_size
        self.in_layer = nn.Linear(configs.patch_size, configs.global_embed_dim)
        self.out_layer = nn.Linear(configs.global_embed_dim, configs.global_embed_dim)
        self.gpt2 = GPT2Model.from_pretrained('./gpt2_offline/', output_attentions=True, output_hidden_states=True)
        self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, X):  #[B, N, M, P]
        x = self.in_layer(X)
        x = rearrange(x, 'b n m p -> (b n) m p')
        out = self.gpt2(inputs_embeds=x).last_hidden_state
        out = rearrange(out, '(b n) m d -> b n m d', b=X.shape[0])
        out = self.out_layer(out)

        return out



