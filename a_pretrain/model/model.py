import math
import torch
import dataclasses
import transformers
import os


tokenizer: transformers.PreTrainedTokenizer = (
    transformers.AutoTokenizer.from_pretrained("a_pretrain/model")
)


class LayerNorm(torch.nn.Module):
    """
    层归一化 (Layer Normalization). （拓展: 更高阶的归一化 RMSNorm）

    将数据分布强制拉回到一个标准状态（例如，均值为0，方差为1），
    对齐不同维度的 “单位”，防止只有对应维度量纲大的参数在有效学习
    """

    weight: torch.nn.Parameter
    bias: torch.nn.Parameter

    def __init__(self, hidden_size: int):
        super().__init__()

        """
        为什么需要可学习的仿射变换参数 (weight/gamma 和 bias/beta)?
        强制标准化是一个“有损”操作：它粗暴地抹去了输入特征原始的均值（偏移）和
        方差（尺度）。然而，这些信息可能对模型的表达能力至关重要。例如，一个特征
        向量的整体“强度/长度” 可能代表了一个 token 的重要性。

        weight (gamma) 和 bias (beta) 给予了网络“反悔”的机会，让它可以学习
        一个最佳的、新的尺度和偏移。这是对表达能力的一种“受控”恢复
        """
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))  # gamma, 初始化为 1
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size))  # beta, 初始化为 0

    def forward(
        self,
        x: torch.Tensor,  # [batch_size, seq_len, hidden_size]
    ) -> torch.Tensor:  # [batch_size, seq_len, hidden_size]
        """
        标准化: 强行将输入的均值变为 0，标准差变为 1（平均距离原点的差距为 1）

        为什么要减去均值? -> 使其“重心”回到原点（“中心化”）。
        这有助于后续层（尤其是激活函数）在其非饱和区域工作
        """
        mean: torch.Tensor = x.mean(dim=-1, keepdim=True)
        var: torch.Tensor = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x: torch.Tensor = (x - mean) / torch.sqrt(
            var + 1e-5
        )  # 1e-5 仅用于防止除零错误

        # 仿射变换: 使用学习到的参数对标准化后的输出进行缩放和偏移
        return (self.weight * norm_x + self.bias).to(x.dtype)


class FeedForward(torch.nn.Module):
    up_proj: torch.nn.Linear
    down_proj: torch.nn.Linear
    dropout: torch.nn.Dropout

    def __init__(
        self,
        hidden_size: int,
        dropout_rate: float,
    ):
        super().__init__()
        intermediate_size = 4 * hidden_size
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(
        self,
        x: torch.Tensor,  # [batch_size, seq_len, hidden_size]
    ) -> torch.Tensor:  # [batch_size, seq_len, hidden_size]
        """
        经典的前馈网络 (升维 -> 非线性 -> 降维) 引入非线性提升模型表达力

        1. up_proj: 扩展特征空间，提供更强的表示能力
        2. relu: max(0, x). 引入非线性，同样提供更强的表示能力。
                 Attention 模块是单纯的线性变换，非线性变换（曲线）能拟合更复杂的情况
        3. dropout: 随机丢弃部分神经元，防止过拟合
        4. down_proj: 将特征投影回原始维度，以便与残差连接相加
        """

        x = self.up_proj(x)  # [batch_size, seq_len, intermediate_size]
        x = torch.nn.functional.relu(
            x
        )  # 扩展思考：relu 有什么问题（提示神经元死亡），因此拓展出了哪些进阶的激活函数？
        x = self.dropout(
            x
        )  # 随机丢弃部分神经元输出。e.g. [0.2, 0.8, -0.5] -> [0.2, 0, -0.5]
        x = self.down_proj(x)  # [batch_size, seq_len, hidden_size]

        return x


@dataclasses.dataclass
class TokensState:
    key: torch.Tensor  # [batch_size, num_heads, past_len, head_dim]
    value: torch.Tensor  # [batch_size, num_heads, past_len, head_dim]


class Attention(torch.nn.Module):
    """
    Self-Attention 架构

    x (B, L_q, H)
    │
    ├───> q_proj ────> Q (B, L_q, H) ───> reshape ───> Q' (B, n_heads, L_q, d_head)
    │
    ├───> k_proj ────> K (B, L_q, H) ───> reshape ───> K' (B, n_heads, L_k, d_head)
    │
    └───> v_proj ────> V (B, L_q, H) ───> reshape ───> V' (B, n_heads, L_k, d_head)


                Q' @ K'^T / sqrt(d_head)
                            │
                        scores (B, n_heads, L_q, L_k)
                            │
                    mask (causal + padding)
                            │
                            softmax
                            │
                    attn_weights (B, n_heads, L_q, L_k)
                            │
                            │ @ V'
                            │
                    mixed_v (B, n_heads, L_q, d_head)
                            │
                        recover_shape
                            │
                    reshaped (B, L_q, H)
                            │
                            o_proj
                            │
                        output (B, L_q, H)

    B: batch_size, L_q: query_seq_len, L_k: key_seq_len, H: hidden_size
    在训练/预填充阶段, L_q == L_k. 在解码阶段, L_q=1, L_k > 1.

    实现优化点（我也还半知半解，留个 TODO，后面接着学习）：
      更强的位置泛化能力 -> RoPE 替代绝对位置编码
      提高内存效率 -> Flash Attention
    """

    num_heads: int
    head_dim: int

    # 矩阵 [hidden_size(row), hidden_size(col)]
    q_proj: torch.nn.Linear
    k_proj: torch.nn.Linear
    v_proj: torch.nn.Linear
    o_proj: torch.nn.Linear
    attn_dropout: torch.nn.Dropout

    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float):
        super().__init__()
        assert (
            hidden_size % num_heads == 0
        ), "hidden_size must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads  # // 整除的意思

        # Q: 为什么不直接 q_proj_for_head1, q_proj_for_head2, ...
        # A: 批处理的思路。为了只做一次计算，而不是分开多个矩阵运算，多次调用 GPU
        self.q_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_dropout = torch.nn.Dropout(dropout_rate)

    def forward(
        self,
        x: torch.Tensor,  # [batch_size, q_len, hidden_size]
        padding_mask: torch.Tensor,  # [batch_size, k_len]; True if padded
        past_kv: TokensState | None = None,
    ) -> tuple[torch.Tensor, TokensState]:
        """
        执行 (Self) Attention 计算, 支持 KV Cache.

        调用约定 (Contract with Caller):
        1. past_kv 用于传递先前计算好的 Key 和 Value (缓存).
           - 训练/Prefill: past_kv 应为 None.
           - 解码: past_kv 应包含来自之前所有步骤的 K/V 状态.

        2. padding_mask 必须反映完整的 Key/Value 序列的填充状态.
           - 令 past_len = past_kv.key.shape[2] (如果 past_kv 不为 None, 否则为 0).
           - 令 q_len = x.shape[1].
           - 则 padding_mask 的形状必须是 [batch_size, past_len + q_len].
           - 调用方有责任在传入前, 将历史的 padding_mask 和当前的 padding_mask 进行拼接.

        Args:
            x: 当前输入的 token embedding.
            padding_mask: 完整的填充掩码.
            past_kv: 历史 K/V 缓存.

        Returns:
            A tuple containing:
            - output (torch.Tensor): 注意力计算的输出, 形状为 [batch_size, q_len, hidden_size].
            - present_kv (TokensState): 更新后的 K/V 缓存, 包含了本次计算的 K/V.
              其内部张量形状为 [batch_size, num_heads, past_len + q_len, head_dim].
        """

        # 投射到 QKV 子空间。[batch_size, num_heads, q_len, head_dim]
        q, k, v = self._mha_project(x)

        """
        思考：
        在 MHA 中，K 和 V 矩阵的“头”数与 Q 矩阵是相同的。
        这意味着 KV Cache 的大小是 [batch_size, num_heads, seq_len, head_dim]。
        随着序列变长，这个 Cache 会变得非常大

        既然瓶颈在于 KV Cache 的大小，而其大小又和“头”数成正比，我们是否真的需要和 Q 一样多的 K 和 V 头？
        Q 的头多，意味着“查询”或“提问”的角度多；K 和 V 的头多，意味着“键值对”的种类多。
        有没有可能，多个不同的“查询”，可以共享同一组“键值对”？
    
        比如，我们能否让 4 个 Query Head 共享同一组 Key Head 和 Value Head？
        提示：Grouped-Query Attention (GQA)
        """
        if past_kv is not None:
            k = torch.cat([past_kv.key, k], dim=2)
            v = torch.cat([past_kv.value, v], dim=2)
        present_kv = TokensState(key=k, value=v)

        # 计算注意力权重。[batch_size, num_heads, seq_len, seq_len]
        attn_weights: torch.Tensor = self._calculate_attn_weights(q, k, padding_mask)

        # 计算并合并多头输出。[batch_size, q_len, hidden_size]
        output = self._merge_output(attn_weights, v)
        return output, present_kv

    def _mha_project(
        self, x: torch.Tensor  # [batch_size, q_len, hidden_size]
    ) -> tuple[
        torch.Tensor,  # q. [batch_size, num_heads, q_len, head_dim]
        torch.Tensor,  # k. [batch_size, num_heads, q_len, head_dim]
        torch.Tensor,  # v. [batch_size, num_heads, q_len, head_dim]
    ]:
        """
        将输入张量 x 投影到 Q, K, V 子空间，并为多头注意力机制重塑其形状

        x (B, L, H)
        │
        ├───> q_proj ────> Q (B, L, H) ───> reshape ───> Q' (B, n_heads, L, d_head)
        │
        ├───> k_proj ────> K (B, L, H) ───> reshape ───> K' (B, n_heads, L, d_head)
        │
        └───> v_proj ────> V (B, L, H) ───> reshape ───> V' (B, n_heads, L, d_head)

        B: batch_size, L: seq_len, H: hidden_size

        思考：为什么我们需要 Q, K, V 三个不同的矩阵？
        为什么自注意力机制（self-attention）不直接使用输入 x 计算，
        而是要先通过 q_proj, k_proj, v_proj 这三个线性变换，将其投影到 Q, K, V 子空间？
        这些线性变换的意义是什么？
        """

        # 线性投影子空间。[batch_size, q_len, hidden_size] -> [batch_size, q_len, hidden_size]
        q: torch.Tensor = self.q_proj.__call__(x)
        k: torch.Tensor = self.k_proj.__call__(x)
        v: torch.Tensor = self.v_proj.__call__(x)

        # 思考：为什么需要将 Q, K, V 进一步拆分成多个“头”（Multi-Head）？
        # 单个“大头”的注意力机制，和多个“小头”的注意力机制，在表达能力上有什么区别？
        #
        # 转换多头 QKV。[batch_size, q_len, hidden_size] -> [batch_size, num_heads, q_len, head_dim]
        batch_size, seq_len, _ = x.shape
        q = self._mha_reshape(q, batch_size, seq_len, self.num_heads, self.head_dim)
        k = self._mha_reshape(k, batch_size, seq_len, self.num_heads, self.head_dim)
        v = self._mha_reshape(v, batch_size, seq_len, self.num_heads, self.head_dim)

        return q, k, v

    def _calculate_attn_weights(
        self,
        q: torch.Tensor,  # [batch_size, num_heads, q_len, head_dim]
        k: torch.Tensor,  # [batch_size, num_heads, k_len, head_dim]
        padding_mask: torch.Tensor,  # [batch_size, k_len]; True if padded
    ) -> torch.Tensor:  # attn_weights. [batch_size, num_heads, q_len, k_len]
        """
        计算注意力权重

                Q' @ K'^T / sqrt(d_head) -> scores (B, n_heads, q_len, k_len)
                            │
                    via mask (causal + padding)
                            │
                    softmax -> attn_weights (B, n_heads, q_len, k_len)
        """

        # 计算注意力分数: Q @ K^T / sqrt(head_dim)。[batch_size, num_heads, q_len, k_len]
        scores: torch.Tensor = torch.matmul(
            q,
            k.transpose(
                -2, -1
            ),  # [batch_size, num_heads, q_len, head_dim] @ [batch_size, num_heads, head_dim, k_len]
        ) / math.sqrt(
            self.head_dim
        )  # 收敛值大小，防止经过 softmax 后，梯度过小

        # 屏蔽因果位和 Padding 位注意力。[batch_size, num_heads, q_len, k_len]
        q_len, k_len = q.shape[-2], k.shape[-2]
        causal_mask: torch.Tensor = self._make_causal_bool_mask(q_len, k_len, q.device)
        masked_scores: torch.Tensor = self._mark_bad_position_to_inf(
            scores, padding_mask, causal_mask
        )

        # 转概率
        return torch.nn.functional.softmax(masked_scores, dim=-1)

    def _merge_output(
        self,
        attn_weights: torch.Tensor,  # [batch_size, num_heads, q_len, k_len]
        v: torch.Tensor,  # [batch_size, num_heads, k_len, head_dim]
    ) -> torch.Tensor:  # [batch_size, q_len, hidden_size]
        """
        attn_weights (B, n_heads, q_len, k_len) @ v (B, n_heads, k_len, d_head) -> mixed_v (B, n_heads, q_len, d_head)
                                │
                        recover_shape -> original_shaped (B, q_len, H)
                                │
                                o_proj -> output (B, q_len, H)

        思考：注意力权重 (attn_weights) 决定了每个 token 对其他 token 的关注程度。
        如果模型在训练时发现，某个 token A 后面总是跟着 token B，
        它可能会学到一种捷径：只要看到 A，就将极高的注意力权重分配给 B。
        这种模式在训练集上表现很好，但在更广泛的数据上可能会失效，这就是过拟合。

        我们如何迫使模型学习更多样、更鲁棒的特征，而不是仅仅依赖于少数几个 token 的强相关性？
        能否在训练过程中，随机地"丢弃"或"忽略"一部分注意力连接，
        从而让模型不能过度依赖任何单一的连接？

        解决方案: 在 attn_weights 上应用 Dropout，随机丢弃一部分注意力连接，
        迫使模型学习更多样化的特征，提升泛化能力
        """

        # 应用 attention dropout 防止过拟合
        attn_weights = self.attn_dropout(attn_weights)

        # 计算加权和。[batch_size, num_heads, q_len, head_dim]
        mixed_v: torch.Tensor = torch.matmul(
            attn_weights,
            v,
        )

        # 恢复矩阵形状。[batch_size, q_len, hidden_size]
        batch_size, _, q_len, _ = mixed_v.shape
        hidden_size = self.num_heads * self.head_dim
        original_shaped: torch.Tensor = (
            mixed_v.transpose(1, 2).contiguous().view(batch_size, q_len, hidden_size)
        )

        # change space
        return self.o_proj(original_shaped)

    def _mha_reshape(
        self,
        x: torch.Tensor,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
    ) -> torch.Tensor:

        # 整型到 [batch_size, seq_len, num_heads, head_dim]
        expanded: torch.Tensor = x.view(batch_size, seq_len, num_heads, head_dim)

        # 转置维度 1 和 2，即进一步整型到 [batch_size, num_heads, seq_len, head_dim]，便于多个 head 间并行计算
        return expanded.transpose(1, 2)

    def _make_causal_bool_mask(
        self, q_len: int, k_len: int, device: torch.device
    ) -> torch.Tensor:  # [q_len, k_len]
        """
        创建因果掩码 (causal mask) 以防止模型看到未来的 token

        支持两种模式:
        1. 训练/预填充: q_len == k_len。创建标准的下三角因果掩码。
        2. 解码: q_len < k_len (通常 q_len=1)。创建的掩码允许查询关注所有过去的键。

        Example (q_len=1, k_len=4):
            [[False, False, False, False]]

        Example (q_len=4, k_len=4):
            [[False,  True,  True,  True],
            [False, False,  True,  True],
            [False, False, False,  True],
            [False, False, False, False]]
        """
        if q_len > k_len:
            raise ValueError(
                f"q_len ({q_len}) > k_len ({k_len}) is not allowed for causal mask."
            )

        past_len = k_len - q_len

        # 1. 允许关注所有历史 token 的部分
        # 这部分掩码全为 False，代表不进行任何遮挡
        past_mask = torch.zeros(q_len, past_len, dtype=torch.bool, device=device)

        # 2. 为当前查询序列本身创建因果掩码的部分
        # 这部分掩码用于防止当前查询序列中的 token 关注其未来的 token
        if q_len > 1:
            # 训练或 Prefill 阶段 (q_len > 1):
            # 创建一个标准的上三角矩阵，用于遮挡未来的 token
            causal_mask_for_q = torch.triu(
                torch.ones(q_len, q_len, dtype=torch.bool, device=device), diagonal=1
            )
        else:
            # 解码阶段 (q_len = 1):
            # 单个 token 没有“未来”可言，所以不需要遮挡任何东西
            causal_mask_for_q = torch.zeros(1, 1, dtype=torch.bool, device=device)

        # 3. 合并两部分掩码
        # 最终掩码允许关注过去，并对现在施加因果约束
        return torch.cat([past_mask, causal_mask_for_q], dim=1)

    def _mark_bad_position_to_inf(
        self,
        scores: torch.Tensor,  # [batch_size, num_heads, q_len, k_len]
        padding_mask: torch.Tensor,  # [batch_size, k_len]; True/False encoding
        causal_mask: torch.Tensor,  # [q_len, k_len]; True/False encoding
    ) -> torch.Tensor:  # [batch_size, num_heads, q_len, k_len]
        """
        扩展 padding_mask 以便与注意力分数矩阵进行广播
        [batch_size, k_len] -> [batch_size, 1, 1, k_len]
        """
        expanded_padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)

        """
        合并因果掩码和填充掩码
        结果形状为 [batch_size, 1, q_len, k_len]
        """
        mask = causal_mask | expanded_padding_mask

        """
        将掩码应用到分数上，被掩盖的位置设置为负无穷大
        Padding 位 & 未来位通常设置为负无穷，经过 softmax 后，权重近似 0
        """
        return scores.masked_fill(mask, -torch.inf)


class Block(torch.nn.Module):
    """
    单个 Transformer Block.

    架构:
       x ────────────────────────────────────────────┐
        │                                            │
        │──> LayerNorm ──> Self-Attention ──> Add ───┤
        │                                            │
        │────────────────────────────────────────────┤
        │                                            │
        │──> LayerNorm ──> FeedForward ──> Add ──────┤
        │                                            │
        └─────────────────────────────────────────────> y
    """

    attn_norm: LayerNorm
    attn: Attention
    ffn_norm: LayerNorm
    ffn: FeedForward
    resid_dropout: torch.nn.Dropout

    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float):
        super().__init__()
        self.attn_norm = LayerNorm(hidden_size)
        self.attn = Attention(
            hidden_size, num_heads, dropout_rate
        )  # wx, 参数 w 的偏导数于输入 x 成正比，所以归一化在变换矩阵之前，稳定梯度
        self.ffn_norm = LayerNorm(hidden_size)
        self.ffn = FeedForward(hidden_size, dropout_rate)
        self.resid_dropout = torch.nn.Dropout(dropout_rate)

    def forward(
        self,
        x: torch.Tensor,  # [batch_size, q_len, hidden_size]
        padding_mask: torch.Tensor,  # [batch_size, k_len]
        past_kv: TokensState | None = None,  # 前面词的 KV Cache
    ) -> tuple[
        torch.Tensor,  # [batch_size, q_len, hidden_size]
        TokensState,  # 包含当前所有词（x）的完整 KV Cache
    ]:
        """
        注意力子层

        Why 残差连接 (y = x + f(x))?
        1. 收敛 f(x) 的语义定位，使得对应的矩阵参数更易训练。
           因为相比不需要残差连接的情况(y = f(x)), f(x) 只需要负责输出 "变化量"，而不是最终的全量输出
        2. 缓解梯度消失。
           在反向传播的过程中，越前面的参数梯度越低，甚至趋近于0，造成梯度消失，再往前的参数无法得到更新
           而残差连接 (y = x + f(x))，相当于绕过 f(x)，建立了一条新连接，能直接避免梯度被 f(x) 所代表的矩阵衰减掉（可以画图，根据链式法则推导下）
        """
        normed_x = self.attn_norm(x)
        delta_from_attn, present_kv = self.attn.__call__(
            normed_x, padding_mask, past_kv
        )
        delta_from_attn = self.resid_dropout(delta_from_attn)
        final_attn_output = x + delta_from_attn

        """
        前馈网络子层
        负责对每个 token 的表示进行非线性变换，增强模型的表达能力
        """
        normed_final_attn_output = self.ffn_norm(final_attn_output)
        delta_from_ffn = self.ffn(normed_final_attn_output)
        delta_from_ffn = self.resid_dropout(delta_from_ffn)
        output = final_attn_output + delta_from_ffn

        return output, present_kv


@dataclasses.dataclass
class Config:
    vocab_size: int = tokenizer.vocab_size
    hidden_size: int = 256  # 复杂度平方增加
    num_layers: int = 12  # 优先增加深度；复杂度线性增加
    num_heads: int = 4  # 与 hidden_size 联动，使得 head_dim=64 (最佳实践)
    max_seq_len: int = 256  # 模型上下文长度。通常 >= 训练集 seq_len
    dropout_rate: float = 0.1  # Dropout 概率，用于防止过拟合


DEFAULT_CONFIG = Config()


class GPT(torch.nn.Module):
    """
    ┌───────────────────┐
    │    input_ids      │  [batch_size, seq_len]
    └───────────────────┘
             │
    ┌────────▼────────┐
    │  _embeddings    │
    │  - tok_embed    │
    │  - pos_embed    │
    └────────┬────────┘
             │
             │ x [batch_size, seq_len, hidden_size]
             │
    ┌────────▼────────┐
    │ _forward_blocks │ ◄── past_kvs (历史 KV Cache)
    │  - Block 1      │
    │  - ...          │
    │  - Block N      │
    └────────┬────────┘
             │
             │ x [batch_size, seq_len, hidden_size]
             │
    ┌────────▼────────┐
    │      norm       │ (LayerNorm)
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │   vocab_proj    │ (Linear: hidden_size -> vocab_size)
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │     logits      │  [batch_size, seq_len, vocab_size]
    └─────────────────┘
    """

    config: Config

    tok_embeddings: torch.nn.Embedding
    pos_embeddings: torch.nn.Embedding
    blocks: torch.nn.ModuleList
    norm: LayerNorm
    vocab_proj: (
        torch.nn.Linear
    )  # 将 hidden_states 投影到词表空间，得到每个位置对下一个 token 的预测
    embd_dropout: torch.nn.Dropout

    def __init__(
        self,
        config: Config | None = DEFAULT_CONFIG,
    ):
        super().__init__()
        self.config = config

        self.tok_embeddings = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embeddings = torch.nn.Embedding(config.max_seq_len, config.hidden_size)
        self.embd_dropout = torch.nn.Dropout(config.dropout_rate)
        self.blocks = torch.nn.ModuleList(
            [
                Block(config.hidden_size, config.num_heads, config.dropout_rate)
                for _ in range(config.num_layers)
            ]
        )
        self.norm = LayerNorm(config.hidden_size)
        self.vocab_proj = torch.nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )
        self.vocab_proj.weight = self.tok_embeddings.weight  # share weight

        print(
            f"模型参数量: {sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6:.2f}M(illion)"
        )
        print(f"上下文窗口: {config.max_seq_len}")

    def forward(
        self,
        input_ids: torch.Tensor,  # [batch_size, q_len]. 输入的 token ID
        padding_mask: torch.Tensor,  # [batch_size, k_len].
        past_kvs: (
            list[TokensState] | None
        ) = None,  # 包含所有 Transformer Block 历史 K/V 缓存的列表
    ) -> tuple[
        torch.Tensor,  # [batch_size, q_len, hidden_size]
        list[TokensState],  # 更新后的所有 K/V 缓存
    ]:
        x, past_kvs = self._embeddings(input_ids, past_kvs)

        blocks_output, present_kvs = self._forward_blocks(x, padding_mask, past_kvs)

        normed = self.norm(blocks_output)
        logits = self.vocab_proj(normed)

        return logits, present_kvs

    def _embeddings(
        self,
        input_ids: torch.Tensor,
        past_kvs: list[TokensState] | None = None,
    ) -> tuple[torch.Tensor, list[TokensState]]:
        # 1. 计算位置信息
        q_len = input_ids.shape[1]
        if past_kvs is not None:
            # 解码模式: K/V 缓存存在, 从上文末尾继续
            past_len = past_kvs[0].key.shape[2]
        else:
            # 训练或预填充模式: K/V 缓存不存在, 从 0 开始
            past_len = 0
            past_kvs = [None] * len(self.blocks)
        pos_start, pos_end = past_len, past_len + q_len

        # 2. 计算嵌入
        token_embeds: torch.Tensor = self.tok_embeddings(input_ids)
        pos_embeds: torch.Tensor = self.pos_embeddings(
            torch.arange(
                start=pos_start, end=pos_end, dtype=torch.long, device=input_ids.device
            )
        )
        embeds: torch.Tensor = token_embeds + pos_embeds

        # 3. 应用 embedding dropout 防止过拟合
        embeds = self.embd_dropout(embeds)

        return embeds, past_kvs

    def _forward_blocks(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        past_kvs: list[TokensState] | None = None,
    ) -> tuple[torch.Tensor, list[TokensState]]:
        present_kvs: list[TokensState] = []
        block: Block
        past_kv: TokensState | None

        for block, past_kv in zip(self.blocks, past_kvs, strict=True):
            x, present_kv = block.__call__(x, padding_mask, past_kv)
            present_kvs.append(present_kv)

        return x, present_kvs

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        checkpoint = {
            "state_dict": self.state_dict(),
            "config": {
                "vocab_size": self.config.vocab_size,
                "hidden_size": self.config.hidden_size,
                "num_layers": self.config.num_layers,
                "num_heads": self.config.num_heads,
                "max_seq_len": self.config.max_seq_len,
                "dropout_rate": self.config.dropout_rate,
            },
        }

        torch.save(checkpoint, filepath)

    @classmethod
    def load(cls, filepath: str, device: str = "cpu") -> "GPT":
        checkpoint = torch.load(filepath, map_location=device)

        config = checkpoint["config"]
        model = cls(
            config=Config(
                vocab_size=config["vocab_size"],
                hidden_size=config["hidden_size"],
                num_layers=config["num_layers"],
                num_heads=config["num_heads"],
                max_seq_len=config["max_seq_len"],
                dropout_rate=config["dropout_rate"],
            )
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)

        return model
