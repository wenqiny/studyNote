import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class Config:
    def __init__(self):
        self.vocab_size = 32000
        self.d_model = 5120
        self.n_layers = 2
        self.n_heads = 8
        self.d_kv_comp = 128
        self.d_rope = 16
        self.n_experts = 32
        self.n_shared = 2
        self.top_k = 2
        self.seq_len = 256
        self.batch_size = 1
        self.ffn_dim = 384
        self.device_groups = 4 # For device-limited routing

config = Config()


class Expert(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.w1 = nn.Linear(conf.d_model, conf.ffn_dim)
        self.w2 = nn.Linear(conf.ffn_dim, conf.d_model)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))
    
class RotaryEmbedding(nn.Module):
    def __init__(self, conf, scale = 40):
        super().__init__()
        assert conf.d_rope % 2 == 0
        self.d_rope = conf.d_rope
        inv_freq = 1 / (10000 ** torch.arange(0, self.d_rope//2, 2).float() / (self.d_rope//2))
        self.register_buffer("inv_freq", inv_freq)
        self.scale = scale

    def forward(self, seq_len):
        t = torch.arange(seq_len, device = self.inv_freq.device).type_as(self.inv_freq) / self.scale
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary(x, cos, sin):
    # split x into two parts, it implys the last shape of x should be double of cos.
    x_rotary, x_base = x.split(cos.shape[-1], dim=-1)
    # there is a element-wise product
    x_rotary = x_rotary * cos + rotate_half(x_rotary) * sin
    return torch.cat((x_rotary, x_base), dim=-1)

class MLA(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.d_head = conf.d_model // conf.n_heads
        self.split_dim = self.d_head - conf.d_rope   
        self.n_heads = conf.n_heads
        self.d_rope = conf.d_rope

        # Projections
        self.W_dkv = nn.Linear(conf.d_model, conf.d_kv_comp)
        self.W_dq = nn.Linear(conf.d_model, conf.d_kv_comp)

        self.W_uk = nn.Linear(conf.d_kv_comp, conf.n_heads * self.split_dim)
        self.W_uv = nn.Linear(conf.d_kv_comp, conf.n_heads * self.d_head)
        self.W_uq = nn.Linear(conf.d_kv_comp, conf.n_heads * self.split_dim)

        self.W_qr = nn.Linear(conf.d_kv_comp, conf.n_heads * conf.d_rope)
        self.W_kr = nn.Linear(conf.d_model, conf.n_heads * conf.d_rope)

        self.rotary = RotaryEmbedding(config)
        self.output = nn.Linear(conf.n_heads * self.d_head, conf.d_model)

    def forward(self, h):
        batch_size, seq_len, _ = h.shape

        # kv compress and decompress
        c_kv = self.W_dkv(h)
        k = self.W_uk(c_kv).view(batch_size, seq_len, self.n_heads, self.split_dim)
        v = self.W_uv(c_kv).view(batch_size, seq_len, self.n_heads, self.d_head)
        k_rot = self.W_kr(h).view(batch_size,  seq_len, self.n_heads, self.d_rope)

        # q compress and decompress
        c_q = self.W_dq(h)
        q_base = self.W_uq(c_q).view(batch_size, seq_len, self.n_heads, self.split_dim)
        q_rot = self.W_qr(c_q).view(batch_size, seq_len, self.n_heads, self.d_rope)

        # prepare for rotary
        rotary_emb = self.rotary(seq_len)
        cos = torch.cos(rotary_emb).view(1, seq_len, 1, -1)
        sin = torch.sin(rotary_emb).view(1, seq_len, 1, -1)

        # apply rotary
        q_rot = apply_rotary(q_rot, cos, sin)
        k_rot = apply_rotary(k_rot, cos, sin)

        # concatenate q and k
        q = torch.cat((q_base, q_rot), dim=-1)
        k = torch.cat((k, k_rot), dim=-1)

        # attention
        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(self.d_head)
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum("bhqk,bkhd->bqhd", attn, v).contiguous().view(batch_size, seq_len, -1)

        return self.output(out), (c_kv, k_rot)


# mla = MLA(config)
# hidden = torch.zeros(2, 128, 5120)
# t = mla(hidden)
