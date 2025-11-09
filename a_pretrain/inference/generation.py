import torch
import transformers
from a_pretrain.model import model


class Generator:
    gpt: model.GPT
    device: str

    def __init__(self, gpt: model.GPT, device: str = "cpu"):
        self.gpt = gpt.to(device).eval()
        self.device = device

    @torch.no_grad()  # 禁止 PyTorch 追踪梯度
    def generate(
        self,
        prompt: str,  # 生产环境通常会批量推理，这里简化成单条
        temperature: float = 0.7,
        top_k: int | None = None,
    ) -> str:
        """
        根据 prompt 生成文本

        Args:
            prompt: 用户的输入文本
            temperature: 温度系数。值越高，生成结果越随机；值越低，越确定。T=0 时等同于贪心搜索。
            top_k: Top-K 采样。仅从概率最高的 K 个 token 中进行采样。如果为 None 则不启用。

        Returns:
            生成的文本结果 (仅包含新生成的部分)
        """

        # 1. 编码与预填充 (Encode & Prefill)
        input_ids: torch.Tensor = self._encode_inputs(prompt)  # [1, prompt_len]
        prompt_len = input_ids.shape[1]

        logits: (
            torch.Tensor
        )  # [1, prompt_len, vocab_size] 下个 Token 的概率得分 for each pos
        present_kvs: list[  # Prefill 指的就是填充 kv cache
            model.TokensState
        ]  # 包含所有 Transformer Block 历史 K/V 缓存的列表
        logits, present_kvs = self.gpt.__call__(
            input_ids,
            padding_mask=torch.zeros_like(input_ids, dtype=torch.bool),
            past_kvs=None,
        )
        last_token_logits = logits[:, -1, :].squeeze(  # [1, vocab_size]
            1
        )  # 对应整个句子的下个词。PS 前面词的计算没有完全浪费，最后一个词向量融合了前面词的计算结果

        # 2. 解码循环
        generated_ids: torch.Tensor = self._decoding_loop(
            last_token_logits[0],
            present_kvs,
            prompt_len,
            temperature,
            top_k,
        )

        return model.tokenizer.decode(
            generated_ids,
            skip_special_tokens=False,  # for debugging
        )

    def _encode_inputs(self, prompt: str) -> torch.Tensor:  # [1, prompt_len]
        tokenized: transformers.tokenization_utils_base.BatchEncoding = (
            model.tokenizer.encode_plus(
                text=prompt,
                return_tensors="pt",
                add_special_tokens=False,
            )
        )

        return tokenized["input_ids"].to(self.device)

    def _decoding_loop(
        self,
        last_token_logits: torch.Tensor,  # [vocab_size]
        present_kvs: list[model.TokensState],
        prompt_len: int,
        temperature: float,
        top_k: int | None,
    ) -> torch.Tensor:  # [generated_len]
        max_seq_len = self.gpt.config.max_seq_len
        generated_token_ids: list[int] = []

        # 循环生成 tokens，直到达到 max_seq_len 或遇到停止 token
        while prompt_len + len(generated_token_ids) < max_seq_len:
            # 采样下一个 token
            next_token_id = self._sample_next_token_id(
                last_token_logits.unsqueeze(0), temperature, top_k
            )[0]
            generated_token_ids.append(next_token_id.item())

            # 检查是否需要停止
            finished = next_token_id.item() == model.tokenizer.eos_token_id
            truncated = prompt_len + len(generated_token_ids) >= max_seq_len
            if finished or truncated:
                break

            # 前向传播获取下一个 token 的 logits
            logits: torch.Tensor
            logits, present_kvs = self.gpt.__call__(  # [1, 1, vocab_size]
                input_ids=next_token_id.unsqueeze(0),  # [1, 1]
                padding_mask=torch.zeros(
                    (1, present_kvs[0].key.shape[2] + 1),
                    dtype=torch.bool,
                    device=self.device,
                ),
                past_kvs=present_kvs,
            )
            last_token_logits = logits.squeeze(1).squeeze(0)  # [vocab_size]

        return torch.tensor(generated_token_ids, device=self.device)

    def _sample_next_token_id(
        self,
        logits: torch.Tensor,  # [1, vocab_size]
        temperature: float,
        top_k: int | None,
    ) -> torch.Tensor:  # [1, 1]
        if temperature == 0:
            # 贪心搜索 (Greedy Search)
            return torch.argmax(logits, dim=-1, keepdim=True)

        # apply temperature
        probs = torch.softmax(  # [1, vocab_size]
            # e.g. temperature>1, 原本概率大的 logit 绝对值衰减更厉害，不同 token 之间的差距变小
            logits / temperature,
            dim=-1,
        )

        # apply top_k
        if top_k is not None:
            probs = self._apply_top_k(probs, top_k)

        # 根据概率抽取 token。[1, vocab_size] -> [1, 1]
        return torch.multinomial(probs, num_samples=1)

    def _apply_top_k(
        self,
        probs: torch.Tensor,  # [1, vocab_size]
        top_k: int,
    ) -> torch.Tensor:  # [1, vocab_size]
        """
        将除了 top_k 之外的 token 概率置为 0
        """

        # 获取第 k 大的概率值
        kth_vals, _ = torch.topk(probs, top_k, dim=-1)  # [1, top_k]
        kth_val = kth_vals[:, -1:]  # [1, 1]

        # 小于 topK 的位置的概率清零
        probs = probs.clone()
        probs.masked_fill_(probs < kth_val, 0.0)

        # 重新归一化，确保概率和为 1
        return probs / probs.sum(dim=-1, keepdim=True)
