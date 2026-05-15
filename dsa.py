import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAtten(nn.Module):
    """标准 Multi-Head Attention — 基类"""

    def __init__(self, hidden_size, num_heads):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.proj_o = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(p=0.1)

        self._build_qkv_projectors()

    def _build_qkv_projectors(self):
        self.proj_q = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.proj_k = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.proj_v = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def _project_qkv(self, x):
        """投影 Q, K, V — 子类覆写此方法即可替换投影策略"""
        batch_size, seq_len, hidden_dim = x.shape
        num_heads, head_dim = self.num_heads, self.head_dim

        q = self.proj_q(x).reshape(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)  # (B,H,S,d)
        k = self.proj_k(x).reshape(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 3, 1)  # (B,H,d,S)
        v = self.proj_v(x).reshape(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)  # (B,H,S,d)
        return q, k, v

    def _compute_score(self, q, k):
        """计算注意力分数 — 子类可覆写以实现解耦 RoPE 等"""
        return q @ k / (self.head_dim ** 0.5)  # (B,H,S,S)

    def _build_mask(self, x):
        """构建 attention mask — 默认无 mask, DSA 覆写此方法"""
        return None

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape

        q, k, v = self._project_qkv(x)
        score = self._compute_score(q, k)

        mask = self._build_mask(x)
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (B,S,S) → (B,1,S,S)
            score = score.masked_fill(~mask, float('-inf'))

        prob = self.attn_dropout(F.softmax(score, dim=-1))
        output = (prob @ v).permute(0, 2, 1, 3).reshape(batch_size, seq_len, hidden_dim)
        return self.proj_o(output)


class MultiHeadLatentAtten(MultiHeadAtten):
    def __init__(self, hidden_size, num_heads, q_lora_rank=0, kv_lora_rank=512, rope_dim=64):
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.rope_dim = rope_dim
        self.nope_dim = hidden_size // num_heads - rope_dim
        super().__init__(hidden_size, num_heads)

    def _build_qkv_projectors(self):
        # --- Query ---
        if self.q_lora_rank > 0:
            self.q_down = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
            self.q_up   = nn.Linear(self.q_lora_rank, self.num_heads * self.nope_dim, bias=False)
        else:
            self.proj_q_nope = nn.Linear(self.hidden_size, self.num_heads * self.nope_dim, bias=False)
        self.proj_q_rope = nn.Linear(self.hidden_size, self.num_heads * self.rope_dim, bias=False)

        # --- KV (低秩压缩) ---
        self.kv_down     = nn.Linear(self.hidden_size, self.kv_lora_rank, bias=False)
        self.k_up        = nn.Linear(self.kv_lora_rank, self.num_heads * self.nope_dim, bias=False)
        self.v_up        = nn.Linear(self.kv_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.proj_k_rope = nn.Linear(self.hidden_size, self.num_heads * self.rope_dim, bias=False)

    def _project_qkv(self, x):
        batch_size, seq_len, _ = x.shape
        num_heads = self.num_heads
        nope_dim, rope_dim, head_dim = self.nope_dim, self.rope_dim, self.head_dim

        # Query: nope + rope 双路
        q_nope = self.q_up(self.q_down(x)) if self.q_lora_rank > 0 else self.proj_q_nope(x)
        q_rope = self.proj_q_rope(x)
        q_nope = q_nope.reshape(batch_size, seq_len, num_heads, nope_dim).permute(0, 2, 1, 3)
        q_rope = q_rope.reshape(batch_size, seq_len, num_heads, rope_dim).permute(0, 2, 1, 3)

        # KV: 压缩 → 恢复
        self.compressed_kv = self.kv_down(x)  # 暂存, DSA 的 _build_mask 要用
        k_nope = self.k_up(self.compressed_kv).reshape(batch_size, seq_len, num_heads, nope_dim).permute(0, 2, 3, 1)
        k_rope = self.proj_k_rope(x).reshape(batch_size, seq_len, num_heads, rope_dim).permute(0, 2, 3, 1)
        v = self.v_up(self.compressed_kv).reshape(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)

        # TODO: apply_rope(q_rope, pos), apply_rope(k_rope, pos)

        return (q_nope, q_rope), (k_nope, k_rope), v

    def _compute_score(self, q, k):
        q_nope, q_rope = q
        k_nope, k_rope = k
        return (q_nope @ k_nope + q_rope @ k_rope) / (self.head_dim ** 0.5)


class DeepSeekSparseAtten(MultiHeadLatentAtten):
    """DSA — 只新增 _build_mask, 其余全部继承 MLA"""

    def __init__(self, hidden_size, num_heads, q_lora_rank=0, kv_lora_rank=512,
                 rope_dim=64, top_k=2048, local_window=256):
        super().__init__(hidden_size, num_heads, q_lora_rank, kv_lora_rank, rope_dim)
        self.top_k = top_k
        self.local_window = local_window

        # 闪电索引器: 在 d_c 维压缩空间上轻量评分
        self.idx_q = nn.Linear(hidden_size, kv_lora_rank, bias=False)
        self.idx_k = nn.Linear(kv_lora_rank, kv_lora_rank, bias=False)

    def _build_mask(self, x):
        """
        ★ DSA 唯一新增逻辑: 闪电索引 → Top-k 选择 → 稀疏 mask
        利用 MLA._project_qkv 中暂存的 self.compressed_kv
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # 闪电索引器评分
        q_idx = self.idx_q(x)                                   # (B, S, d_c)
        k_idx = self.idx_k(self.compressed_kv)                  # (B, S, d_c)
        scores = F.relu(q_idx @ k_idx.transpose(-1, -2) / (self.kv_lora_rank ** 0.5))

        # 因果 mask
        causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        scores = scores.masked_fill(~causal.unsqueeze(0), float('-inf'))

        # Top-k 选择
        _, topk_indices = scores.topk(min(self.top_k, seq_len), dim=-1)
        topk_mask = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bool, device=device)
        topk_mask.scatter_(-1, topk_indices, True)

        # 局部窗口
        pos = torch.arange(seq_len, device=device)
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)
        local_mask = (0 <= diff) & (diff < self.local_window)  # (S, S)

        return (topk_mask | local_mask.unsqueeze(0)) & causal.unsqueeze(0)  # (B, S, S)


if __name__ == "__main__":
    B, S, D, H = 2, 512, 1024, 8
    x = torch.randn(B, S, D)

    for name, model in [
        ("MHA", MultiHeadAtten(D, H)),
        ("MLA", MultiHeadLatentAtten(D, H, q_lora_rank=256, kv_lora_rank=128, rope_dim=32)),
        ("DSA", DeepSeekSparseAtten(D, H, q_lora_rank=256, kv_lora_rank=128, rope_dim=32, top_k=64, local_window=32)),
    ]:
        out = model(x)
        n = sum(p.numel() for p in model.parameters())
        print(f"{name}  output={tuple(out.shape)}  params={n:,}")
