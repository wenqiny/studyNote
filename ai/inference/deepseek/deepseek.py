import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import math

class Config:
    def __init__(self):
        self.vocab_size = 16000
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

class DeepSeekMoE(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.shared_experts = nn.ModuleList([Expert(conf) for _ in range(conf.n_shared)])
        self.routed_experts = nn.ModuleList([Expert(conf) for _ in range(conf.n_experts)])
        self.gate = nn.Linear(conf.d_model, conf.n_experts)
        self.aux_loss = 0.0
        self.top_k = conf.top_k
        self.n_experts = conf.n_experts
    
    def forward(self, x):
        # x [batch, seq_len, d_model]

        # shared_out [batch, seq_len, d_model]
        shared_out = sum(expert(x) for expert in self.shared_experts)

        # router
        routed_logits = self.gate(x)
        # probs [batch, seq_len, n_experts]
        probs = F.softmax(routed_logits, dim=-1)
        # topk_probs, topk_indices [batch, seq_len, top_k]
        topk_probs, topk_indices = probs.topk(self.top_k, dim=-1)

        # static execution time for each expert
        # expert_count [n_experts]
        expert_count = torch.zeros(self.n_experts, device=x.device)
        expert_count.scatter_add_(0, topk_indices.view(-1), torch.ones_like(topk_indices.view(-1), dtype=torch.float))
        self.aux_loss += expert_count.float().var() * 0.003

        # get routed output
        routed_out = torch.zeros_like(x)
        for k in range(self.top_k):
            # expert_mask [batch, seq_len]
            expert_mask = topk_indices[..., k]
            expert_contrib = torch.zeros_like(x)

            for expert_idx in range(self.n_experts):
                # mask [batch, seq_len]
                mask = (expert_mask == expert_idx)
                if mask.any():
                    # x[mask].shape [selected token, d_model]
                    # expert_out [selected token, d_model]
                    expert_out = self.routed_experts[expert_idx](x[mask])
                    expert_contrib[mask] = expert_out * topk_probs[..., k][mask].unsqueeze(-1)
            routed_out += expert_contrib

        return routed_out + shared_out
    

# moe = DeepSeekMoE(config)
# hidden = torch.zeros(2, 128, 5120)
# t = moe(hidden)
# print(t.shape)

class TransformerBlock(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.norm1 = nn.LayerNorm(conf.d_model)
        self.attn = MLA(conf)
        self.norm2 = nn.LayerNorm(conf.d_model)
        self.moe = DeepSeekMoE(config)

    def forward(self, x):
        attn_out, c_kv = checkpoint(self.attn, self.norm1(x))

        x = x + attn_out
        moe_out = checkpoint(self.moe, self.norm2(x))

        x = x + moe_out

        return x, c_kv
    
class DeepSeekV2(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.embed = nn.Embedding(conf.vocab_size, conf.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(conf) for _ in range(conf.n_layers)])
        self.norm = nn.LayerNorm(conf.d_model)
        self.lm_head = nn.Linear(conf.d_model, conf.vocab_size)

        # Better initialization with residual scaling
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p, gain=0.1/math.sqrt(config.n_layers))
        # Add residual scaling
        for block in self.blocks:
            block.attn.output.weight.data.mul_(0.1)
            block.moe.shared_experts[0].w2.weight.data.mul_(0.1)
    
    def forward(self, input_ids):
        x = self.embed(input_ids)
        total_aux_loss = 0.0

        for block in self.blocks:
            x, _ = block(x)
            total_aux_loss += block.moe.aux_loss

        return self.lm_head(self.norm(x)), total_aux_loss
    
def train():
    model = DeepSeekV2(config).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

    # Learning rate schedule with warmup
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-4,  
        total_steps=40,
        pct_start=0.1, 
    )

    for epoch in range(40):
        # Use structured inputs
        inputs = torch.randint(0, config.vocab_size // 10,
                               (config.batch_size, config.seq_len + 1)).cuda()

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits, aux_loss = model(inputs[:, :-1])
            loss = F.cross_entropy(logits.view(-1, config.vocab_size),
                                   inputs[:, 1:].contiguous().view(-1))
            loss += 0.0001 * aux_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        lr_scheduler.step()

        print(f"Epoch {epoch} Loss: {loss.item():.4f}")


if __name__ == "__main__":
    train()