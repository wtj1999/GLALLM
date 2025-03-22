import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from models.Local_Branch import STHGNN
from models.Global_Branch import LLM4TS
from models.STA_Fusion import STA_Fusion

class GLALLM(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1
        self.global_branch = LLM4TS(configs)
        self.local_branch = STHGNN(configs)
        self.sta_fusion = STA_Fusion(configs)
        self.out_layer = nn.Linear(configs.fused_embed_dim * self.patch_num, configs.pred_len)




    def forward(self, X, X_mark):
        B, T, N = X.shape
        x = rearrange(X, 'b l n -> b n l')
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        global_embed = self.global_branch(x)
        local_embed = self.local_branch(x[:, :, -1, :].transpose(1, 2)).transpose(1, 2)
        fused_embed = self.sta_fusion(global_embed, local_embed)
        out = self.out_layer(fused_embed.reshape(B, N, -1)).transpose(1, 2)

        return out



