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
        nb, ns, nd = x.shape
        nh, dh = self.num_heads, self.head_dim

        q = self.proj_q(x).reshape(nb, ns, nh, dh).permute(0, 2, 1, 3)  # (B,H,S,d)
        k = self.proj_k(x).reshape(nb, ns, nh, dh).permute(0, 2, 3, 1)  # (B,H,d,S)
        v = self.proj_v(x).reshape(nb, ns, nh, dh).permute(0, 2, 1, 3)  # (B,H,S,d)
        return q, k, v

    def _compute_score(self, q, k):
        """计算注意力分数 — 子类可覆写以实现解耦 RoPE 等"""
        return q @ k / (self.head_dim ** 0.5)              # (B,H,S,S)

    def _build_mask(self, x):
        """构建 attention mask — 默认无 mask, DSA 覆写此方法"""
        return None

    def forward(self, x):
        nb, ns, nd = x.shape

        q, k, v = self._project_qkv(x)
        score = self._compute_score(q, k)

        mask = self._build_mask(x)
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)            # (B,S,S) → (B,1,S,S)
            score = score.masked_fill(~mask, float('-inf'))

        prob = self.attn_dropout(F.softmax(score, dim=-1))
        y = (prob @ v).permute(0, 2, 1, 3).reshape(nb, ns, nd)
        return self.proj_o(y)


class MultiHeadLatentAtten(MultiHeadAtten):
    """MLA — 覆写 _project_qkv 和 _compute_score, 其余继承"""

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

    def _get_c_kv(self, x):
        """计算 KV 压缩表示 — DSA 子类也会复用"""
        return self.kv_down(x)                             # (B, S, d_c)

    def _project_qkv(self, x):
        nb, ns, _ = x.shape
        nh = self.num_heads
        d_nope, d_rope, dh = self.nope_dim, self.rope_dim, self.head_dim

        # Query
        q_nope = self.q_up(self.q_down(x)) if self.q_lora_rank > 0 else self.proj_q_nope(x)
        q_rope = self.proj_q_rope(x)
        q_nope = q_nope.reshape(nb, ns, nh, d_nope).permute(0, 2, 1, 3)
        q_rope = q_rope.reshape(nb, ns, nh, d_rope).permute(0, 2, 1, 3)

        # KV (压缩 → 恢复)
        self._c_kv = self._get_c_kv(x)                    # 暂存, DSA 的 _build_mask 要用
        k_nope = self.k_up(self._c_kv).reshape(nb, ns, nh, d_nope).permute(0, 2, 3, 1)
        k_rope = self.proj_k_rope(x).reshape(nb, ns, nh, d_rope).permute(0, 2, 3, 1)
        v = self.v_up(self._c_kv).reshape(nb, ns, nh, dh).permute(0, 2, 1, 3)

        # TODO: apply_rope(q_rope, pos), apply_rope(k_rope, pos)

        # 把 nope 和 rope 打包成元组, 传给 _compute_score 解包
        q = (q_nope, q_rope)
        k = (k_nope, k_rope)
        return q, k, v

    def _compute_score(self, q, k):
        q_nope, q_rope = q
        k_nope, k_rope = k
        return (q_nope @ k_nope + q_rope @ k_rope) / (self.head_dim ** 0.5)


class DeepSeekSparseAtten(MultiHeadLatentAtten):
    """DSA — 只需覆写 _build_mask, 其余全部继承 MLA"""

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
        利用 MLA._project_qkv 中暂存的 self._c_kv
        """
        B, S, _ = x.shape
        device = x.device

        # --- 闪电索引器评分 ---
        q_idx = self.idx_q(x)                             # (B, S, d_c)
        k_idx = self.idx_k(self._c_kv)                    # (B, S, d_c)
        scores = F.relu(q_idx @ k_idx.transpose(-1, -2) / (self.kv_lora_rank ** 0.5))

        # --- 因果 mask ---
        causal = torch.tril(torch.ones(S, S, dtype=torch.bool, device=device))
        scores = scores.masked_fill(~causal.unsqueeze(0), float('-inf'))

        # --- Top-k 选择 ---
        k = min(self.top_k, S)
        _, topk_idx = scores.topk(k, dim=-1)
        topk_mask = torch.zeros(B, S, S, dtype=torch.bool, device=device)
        topk_mask.scatter_(-1, topk_idx, True)

        # --- 局部窗口 ---
        pos = torch.arange(S, device=device)
        local = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs().le(self.local_window - 1) & causal

        # --- 合并 ---
        return (topk_mask | local.unsqueeze(0)) & causal.unsqueeze(0)  # (B, S, S)


# =================================================================
# 验证: 三个类共享同一个 forward, 输出维度一致
# =================================================================
if __name__ == "__main__":
    B, S, D, H = 2, 512, 1024, 8
    x = torch.randn(B, S, D)

    configs = [
        ("MHA", MultiHeadAtten(D, H), {}),
        ("MLA", MultiHeadLatentAtten(D, H, q_lora_rank=256, kv_lora_rank=128, rope_dim=32), {}),
        ("DSA", DeepSeekSparseAtten(D, H, q_lora_rank=256, kv_lora_rank=128, rope_dim=32,
                                     top_k=64, local_window=32), {}),
    ]

    for name, model, _ in configs:
        out = model(x)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{name}  output={tuple(out.shape)}  params={n_params:,}")

    # 继承关系验证
    dsa = configs[2][1]
    print(f"\nDSA isinstance MLA: {isinstance(dsa, MultiHeadLatentAtten)}")
    print(f"DSA isinstance MHA: {isinstance(dsa, MultiHeadAtten)}")
    print(f"DSA.forward is MHA.forward: {type(dsa).forward is MultiHeadAtten.forward}")
