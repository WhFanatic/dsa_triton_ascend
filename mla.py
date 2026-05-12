import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAtten(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.proj_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_o = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(p=0.1)
        self.hidden_size = hidden_size
        self.num_heads = num_heads

    def forward(self, x):
        # x (B, S, D)
        nb, ns, nd = x.shape
        nh = self.num_heads
        dh = nd // nh
        scale = dh**.5

        q = self.proj_q(x).reshape(nb, ns, nh, dh).permute(0, 2, 1, 3) # (B, H, S, d)
        k = self.proj_k(x).reshape(nb, ns, nh, dh).permute(0, 2, 3, 1) # (B, H, d, S)
        v = self.proj_v(x).reshape(nb, ns, nh, dh).permute(0, 2, 1, 3) # (B, H, S, d)
        
        prob = F.softmax(q @ k / scale, dim=-1) # (B, H, S, S)
        prob = self.attn_dropout(prob)
        y = (prob @ v).permute(0, 2, 1, 3).reshape(nb, ns, nd) # (B, S, D)
        return self.proj_o(y)


class MultiHeadLatentAtten(nn.Module):
    """
    Multi-head Latent Attention (MLA) — DeepSeek-V2/V3 风格
    
    核心思路：用低秩压缩替代完整的 KV 缓存
      MHA:  h → W_K → K (H·d_h 维)   缓存 K
            h → W_V → V (H·d_h 维)   缓存 V
      MLA:  h → W_DKV → c^KV (d_c 维) 只缓存 c^KV
            c^KV → W_UK → K           按需恢复
            c^KV → W_UV → V           按需恢复
    
    同样对 Query 做低秩压缩以节省训练时激活内存。
    解耦 RoPE: 额外用少量维度 (d_rope) 携带位置信息。
    """

    def __init__(
        self,
        hidden_size: int,       # 模型维度 d (对应你 MHA 的 hidden_size)
        num_heads: int,         # 注意力头数 H
        q_lora_rank: int = 0,   # Query 压缩维度 (0 = 不压缩 Query)
        kv_lora_rank: int = 512,# KV 压缩维度 d_c (核心参数)
        rope_dim: int = 64,     # 解耦 RoPE 维度 d_rope
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads  # d_h
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.rope_dim = rope_dim
        # 每个 head 中参与"内容注意力"的维度 (去掉 rope 占用的部分)
        self.nope_dim = self.head_dim - rope_dim

        # ============================================================
        # Query 投影
        # ============================================================
        if q_lora_rank > 0:
            # 压缩路线: h → c^Q (低维) → Q_nope (内容)
            self.q_down = nn.Linear(hidden_size, q_lora_rank, bias=False)
            self.q_up   = nn.Linear(q_lora_rank, num_heads * self.nope_dim, bias=False)
        else:
            # 不压缩 Query (简化版, 对标你的 proj_q)
            self.proj_q_nope = nn.Linear(hidden_size, num_heads * self.nope_dim, bias=False)

        # Query 的 RoPE 分支 (始终独立投影, 不经过压缩)
        self.proj_q_rope = nn.Linear(hidden_size, num_heads * rope_dim, bias=False)

        # ============================================================
        # KV 投影 (MLA 核心: 下投影 + 上投影)
        # ============================================================
        # 下投影: h → c^KV (d_c 维, 这个被缓存)
        self.kv_down = nn.Linear(hidden_size, kv_lora_rank, bias=False)
        # 上投影: c^KV → K_nope (内容 Key)
        self.k_up    = nn.Linear(kv_lora_rank, num_heads * self.nope_dim, bias=False)
        # 上投影: c^KV → V
        self.v_up    = nn.Linear(kv_lora_rank, num_heads * self.head_dim, bias=False)
        # Key 的 RoPE 分支 (从 h 直接投影, 不经过压缩)
        self.proj_k_rope = nn.Linear(hidden_size, num_heads * rope_dim, bias=False)

        # ============================================================
        # 输出投影 (和你的 proj_o 一样)
        # ============================================================
        self.proj_o = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(p=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, S, D)  与你 MHA 的输入完全一致
        """
        nb, ns, nd = x.shape
        nh = self.num_heads
        dh = self.head_dim
        d_nope = self.nope_dim
        d_rope = self.rope_dim

        # ===========================================================
        # 1. Query: 内容分支 + RoPE 分支
        # ===========================================================
        if self.q_lora_rank > 0:
            q_nope = self.q_up(self.q_down(x))          # (B, S, H*d_nope)
        else:
            q_nope = self.proj_q_nope(x)                # (B, S, H*d_nope)

        q_rope = self.proj_q_rope(x)                    # (B, S, H*d_rope)

        # reshape 成多头格式 (和你 MHA 的 reshape+permute 一样)
        q_nope = q_nope.reshape(nb, ns, nh, d_nope).permute(0, 2, 1, 3) # (B,H,S,d_nope)
        q_rope = q_rope.reshape(nb, ns, nh, d_rope).permute(0, 2, 1, 3) # (B,H,S,d_rope)

        # TODO: 对 q_rope 施加 RoPE 旋转 (此处省略, 实际需要 apply_rope(q_rope, pos))

        # ===========================================================
        # 2. KV: 下投影压缩 → 上投影恢复 (MLA 核心)
        # ===========================================================
        # ★ 这一步是和 MHA 最大的区别:
        #   MHA: k = self.proj_k(x)  直接得到完整 K
        #   MLA: 先压缩到 d_c 维, 推理时只缓存这个
        c_kv = self.kv_down(x)                          # (B, S, d_c=512) ← 缓存这个!

        k_nope = self.k_up(c_kv)                        # (B, S, H*d_nope) 按需恢复
        v      = self.v_up(c_kv)                        # (B, S, H*d_h)  按需恢复
        k_rope = self.proj_k_rope(x)                    # (B, S, H*d_rope) 位置 Key

        # reshape 成多头格式
        k_nope = k_nope.reshape(nb, ns, nh, d_nope).permute(0, 2, 3, 1)  # (B,H,d_nope,S) 注意转置
        k_rope = k_rope.reshape(nb, ns, nh, d_rope).permute(0, 2, 3, 1)# (B,H,d_rope,S)
        v      = v.reshape(nb, ns, nh, dh).permute(0, 2, 1, 3)         # (B,H,S,d_h)

        # TODO: 对 k_rope 施加 RoPE 旋转 (省略)

        # ===========================================================
        # 3. 分别计算内容注意力和位置注意力, 求和
        # ===========================================================
        #   MHA:  score = q @ k / scale          一次搞定
        #   MLA:  score = q_nope @ k_nope        内容匹配 (可以做权重吸收优化)
        #                + q_rope @ k_rope        位置匹配
        scale = dh ** 0.5
        score_nope = q_nope @ k_nope              # (B,H,S,S) 内容注意力
        score_rope = q_rope @ k_rope              # (B,H,S,S) 位置注意力
        score = (score_nope + score_rope) / scale

        prob = F.softmax(score, dim=-1)
        prob = self.attn_dropout(prob)

        # ===========================================================
        # 4. 聚合 + 输出投影 (和你的 MHA 完全一样)
        # ===========================================================
        y = (prob @ v).permute(0, 2, 1, 3).reshape(nb, ns, nd)  # (B,S,D)
        return self.proj_o(y)


class DeepSeekSparseAtten(nn.Module):
    """
    DeepSeek Sparse Attention (DSA) — 在你的 MLA 基础上加入稀疏选择
 
    和你的 MLA 的关系:
      MLA:  对所有 S 个历史 Token 做 Attention (仍是 O(S²))
      DSA:  用轻量索引器给所有历史 Token 打分 → 只选 Top-k 个 → 在 k 个上做 MLA
 
    新增组件:
      1. Lightning Indexer  — 复用 c^KV 压缩表示, 快速评分
      2. Token Selector     — Top-k 选择 + 稀疏 mask 构建
      3. Local Window       — 始终保留最近 w 个 Token (防止遗漏近邻)
    """
 
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        q_lora_rank: int = 0,
        kv_lora_rank: int = 512,
        rope_dim: int = 64,
        # ---- DSA 新增参数 ----
        top_k: int = 2048,       # 从全局选多少个 Token
        local_window: int = 256, # 始终保留的局部窗口大小
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.rope_dim = rope_dim
        self.nope_dim = self.head_dim - rope_dim
        self.top_k = top_k
        self.local_window = local_window
 
        # ============================================================
        # Query 投影 (和你的 MLA 完全一样)
        # ============================================================
        if q_lora_rank > 0:
            self.q_down = nn.Linear(hidden_size, q_lora_rank, bias=False)
            self.q_up   = nn.Linear(q_lora_rank, num_heads * self.nope_dim, bias=False)
        else:
            self.proj_q_nope = nn.Linear(hidden_size, num_heads * self.nope_dim, bias=False)
 
        self.proj_q_rope = nn.Linear(hidden_size, num_heads * rope_dim, bias=False)
 
        # ============================================================
        # KV 投影 (和你的 MLA 完全一样)
        # ============================================================
        self.kv_down     = nn.Linear(hidden_size, kv_lora_rank, bias=False)
        self.k_up        = nn.Linear(kv_lora_rank, num_heads * self.nope_dim, bias=False)
        self.v_up        = nn.Linear(kv_lora_rank, num_heads * self.head_dim, bias=False)
        self.proj_k_rope = nn.Linear(hidden_size, num_heads * rope_dim, bias=False)
 
        # ============================================================
        # ★ DSA 新增: Lightning Indexer (闪电索引器)
        # ============================================================
        # 直接在 c^KV 的压缩空间 (d_c 维) 上做轻量评分
        # 索引器的 Query 投影: h → d_c 维 (复用压缩维度, 不走完整 H*d_h)
        self.idx_q = nn.Linear(hidden_size, kv_lora_rank, bias=False)
        # 索引器的 Key 投影: c^KV → d_c 维 (c^KV 本身就是 d_c 维, 用线性变换增加表达力)
        self.idx_k = nn.Linear(kv_lora_rank, kv_lora_rank, bias=False)
 
        # ============================================================
        # 输出投影 (和你的 MLA 完全一样)
        # ============================================================
        self.proj_o = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(p=0.1)
 
    def _lightning_index(
        self,
        x: torch.Tensor,     # (B, S, D) 输入
        c_kv: torch.Tensor,   # (B, S, d_c) 所有 Token 的 KV 压缩表示
    ) -> torch.Tensor:
        """
        闪电索引器: 为每个 Query 位置对所有历史 Key 位置打分
 
        和完整 Attention 的区别:
          完整:  Q(H*d_h) × K(H*d_h)^T  →  维度大, 按 Head 分组
          索引器: q_idx(d_c) × k_idx(d_c)^T  →  维度小, 跨 Head 共享
 
        返回: (B, S, S) 的相关性分数 (所有 Head 共享同一份分数)
        """
        q_idx = self.idx_q(x)          # (B, S, d_c)
        k_idx = self.idx_k(c_kv)       # (B, S, d_c)
 
        # 点积评分, 缩放防止数值过大
        scale = self.kv_lora_rank ** 0.5
        scores = q_idx @ k_idx.transpose(-1, -2) / scale  # (B, S, S)
 
        # ReLU 激活: 大部分分数归零 → 天然稀疏
        scores = F.relu(scores)
        return scores
 
    def _select_tokens(
        self,
        scores: torch.Tensor,  # (B, S, S) 索引器分数
    ) -> torch.Tensor:
        """
        Token 选择器: 基于索引器分数构建稀疏 mask
 
        策略: Top-k 全局选择 ∪ 局部窗口 ∪ 因果 mask
 
        返回: (B, S, S) 的 bool mask, True = 允许 attend
        """
        B, S, _ = scores.shape
        device = scores.device
 
        # --- 因果 mask: 只能看到当前及之前的 Token ---
        causal = torch.tril(torch.ones(S, S, dtype=torch.bool, device=device))  # (S, S)
 
        # --- 局部窗口: 始终保留最近 w 个 ---
        row_idx = torch.arange(S, device=device).unsqueeze(1)  # (S, 1)
        col_idx = torch.arange(S, device=device).unsqueeze(0)  # (1, S)
        local = (row_idx - col_idx >= 0) & (row_idx - col_idx < self.local_window)  # (S, S)
 
        # --- Top-k 全局选择 ---
        # 把未来位置 (因果 mask 之外) 的分数设为 -inf, 防止选到
        scores_masked = scores.masked_fill(~causal.unsqueeze(0), float('-inf'))
 
        # 对每个 Query 位置, 从所有合法历史位置中选 Top-k
        k = min(self.top_k, S)  # 序列太短时退化为 full attention
        _, topk_idx = scores_masked.topk(k, dim=-1)  # (B, S, k)
 
        # 把 Top-k 索引转成 bool mask
        topk_mask = torch.zeros(B, S, S, dtype=torch.bool, device=device)
        topk_mask.scatter_(-1, topk_idx, True)
 
        # --- 合并: 局部窗口 ∪ Top-k, 再与因果 mask 取交集 ---
        final_mask = (local.unsqueeze(0) | topk_mask) & causal.unsqueeze(0)
 
        return final_mask
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, S, D)
 
        整体流程:
          1. 计算 c^KV (和你的 MLA 一样)
          2. ★ 闪电索引器 + Token 选择 → 稀疏 mask
          3. 从 c^KV 恢复 K, V (和你的 MLA 一样)
          4. 用稀疏 mask 做 Attention (只改了 mask, 其余和你的 MLA 一样)
        """
        nb, ns, nd = x.shape
        nh = self.num_heads
        dh = self.head_dim
        d_nope = self.nope_dim
        d_rope = self.rope_dim
 
        # ===========================================================
        # 1. KV 下投影 (和你的 MLA 完全一样)
        #    ★ 注意这里提前算 c_kv, 因为索引器要用
        # ===========================================================
        c_kv = self.kv_down(x)                          # (B, S, d_c)
 
        # ===========================================================
        # 2. ★ DSA 独有: 闪电索引 + Token 选择
        # ===========================================================
        idx_scores = self._lightning_index(x, c_kv)     # (B, S, S)
        sparse_mask = self._select_tokens(idx_scores)   # (B, S, S) bool
 
        # ===========================================================
        # 3. Query 投影 (和你的 MLA 完全一样)
        # ===========================================================
        if self.q_lora_rank > 0:
            q_nope = self.q_up(self.q_down(x))
        else:
            q_nope = self.proj_q_nope(x)
 
        q_rope = self.proj_q_rope(x)
 
        q_nope = q_nope.reshape(nb, ns, nh, d_nope).permute(0, 2, 1, 3)
        q_rope = q_rope.reshape(nb, ns, nh, d_rope).permute(0, 2, 1, 3)
 
        # TODO: apply_rope(q_rope, pos)
 
        # ===========================================================
        # 4. KV 上投影 + RoPE Key (和你的 MLA 完全一样)
        # ===========================================================
        k_nope = self.k_up(c_kv)
        v      = self.v_up(c_kv)
        k_rope = self.proj_k_rope(x)
 
        k_nope = k_nope.reshape(nb, ns, nh, d_nope).permute(0, 2, 3, 1)
        k_rope = k_rope.reshape(nb, ns, nh, d_rope).permute(0, 2, 3, 1)
        v      = v.reshape(nb, ns, nh, dh).permute(0, 2, 1, 3)
 
        # TODO: apply_rope(k_rope, pos)
 
        # ===========================================================
        # 5. Attention 计算 (和你的 MLA 唯一区别: 加了稀疏 mask)
        # ===========================================================
        scale = dh ** 0.5
        score_nope = q_nope @ k_nope                    # (B, H, S, S)
        score_rope = q_rope @ k_rope                    # (B, H, S, S)
        score = (score_nope + score_rope) / scale
 
        # ★ DSA 核心: 用稀疏 mask 遮盖未选中的 Token
        #   sparse_mask: (B, S, S) → (B, 1, S, S) 广播到所有 Head
        score = score.masked_fill(~sparse_mask.unsqueeze(1), float('-inf'))
 
        prob = F.softmax(score, dim=-1)
        prob = self.attn_dropout(prob)
 
        # ===========================================================
        # 6. 聚合 + 输出 (和你的 MLA 完全一样)
        # ===========================================================
        y = (prob @ v).permute(0, 2, 1, 3).reshape(nb, ns, nd)
        return self.proj_o(y)


# =================================================================
# 验证: 维度对齐, 前向传播通过
# =================================================================
if __name__ == "__main__":
    B, S, D, H = 2, 128, 1024, 8

    mha = MultiHeadLatentAtten(
        hidden_size=D,
        num_heads=H,
        q_lora_rank=256,    # Query 也做压缩
        kv_lora_rank=128,   # KV 压缩到 128 维
        rope_dim=32,        # RoPE 占 32 维
    )

    x = torch.randn(B, S, D)
    out = mha(x)
    print(f"Input:  {x.shape}")     # (2, 128, 1024)
    print(f"Output: {out.shape}")   # (2, 128, 1024)

    # 对比缓存大小
    full_kv_cache  = 2 * H * (D // H) * S   # MHA: 2 * K + V
    mla_cache      = 128 * S + H * 32 * S    # MLA: c^KV + k_rope
    print(f"\nMHA KV cache per sample: {full_kv_cache:,} elements")
    print(f"MLA    cache per sample: {mla_cache:,} elements")
    print(f"Compression ratio:       {full_kv_cache / mla_cache:.1f}x")