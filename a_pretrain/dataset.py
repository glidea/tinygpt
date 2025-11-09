import torch
import torch.utils.data
import transformers
import datasets
import torch.utils.data.sampler
from a_pretrain.model import model

# 数据集：https://huggingface.co/datasets/roneneldan/TinyStories
NAME = "roneneldan/TinyStories"
TRAIN_SUBSET = "train[:97%]"  # 训练集
VAL_SUBSET = "train[97%:]"  # 验证集
BATCH_SIZE = 16
SEQ_LEN = 256
LOSS_IGNORE_LABEL = -100  # 不计算该位置损失


def train_stream() -> torch.utils.data.DataLoader:
    return _build_dataloader(subset=TRAIN_SUBSET, batch_size=BATCH_SIZE, shuffle=True)


def validation_stream() -> torch.utils.data.DataLoader:
    return _build_dataloader(subset=VAL_SUBSET, batch_size=BATCH_SIZE, shuffle=False)


def _build_dataloader(
    subset: str, batch_size: int, shuffle: bool
) -> torch.utils.data.DataLoader:
    """
    datasets.Dataset 可以理解为原始表格数据（HuggingFace 上喽一眼文件内容就有实感了），
    同时提供一系列 ETL 接口，比如 map, filter, batch, etc.

    思考：流式加载 vs 一次性加载
    当前实现是一次性加载整个数据集到内存（虽然用了 mmap，但 map 操作会处理所有数据）。
    对于超大规模数据集（TB 级别），这种方式可能不太合适。

    提示：datasets.IterableDataset + streaming=True
    """
    print(f"Loading {subset} dataset from HuggingFace...")
    raw: datasets.Dataset = datasets.load_dataset(
        path=NAME,
        split=subset,
    )
    print(f"{subset} dataset loaded from HuggingFace.")
    print(f"{subset} dataset size:", len(raw))
    print(f"{subset} dataset example:", raw[0])
    print(
        f"{subset} dataset example token count:",
        len(model.tokenizer.encode(raw[0]["text"])),
    )

    # 准备 Input IDs & Attention Mask
    print(
        'Preparing Input IDs & Attention Mask... e.g. "你好，世界" -> "你 好 ， 世 界 [EOS] [PAD] [PAD]" -> [872, 1962, ..., 1, 0, 0]'
    )
    tokenized: datasets.Dataset = raw.map(
        function=_tokenize,
        batched=True,
        num_proc=4,
        remove_columns=raw.column_names,  # 避免新生成的 tokenized 包含 raw 的数据列
    )
    print("Preparing Input IDs & Attention Mask done.")
    print("Preparing Input IDs & Attention Mask example:", tokenized[0])

    # 准备目标值 labels
    print(
        'Preparing target labels... e.g. "[872, 1962, ..., 1]" -> "[1962, ..., 1, -100]"'
    )
    ds: datasets.Dataset = tokenized.map(
        function=_append_target_labels,
        batched=True,
        num_proc=4,
    )
    ds.set_format("torch")
    print("Preparing target labels done.")
    print("Preparing target labels example:", ds[0])

    # build 数据集的流式迭代器
    sampler: torch.utils.data.sampler.Sampler = (
        torch.utils.data.sampler.RandomSampler(ds)
        if shuffle
        else torch.utils.data.sampler.SequentialSampler(ds)
    )

    return torch.utils.data.DataLoader(
        dataset=ds,
        # 随机采样（训练集）或顺序采样（验证集）
        # 验证集使用顺序采样，保证每次验证结果一致，便于比较
        sampler=sampler,
        # 一次性采样多少个 Simples
        #  较大值：
        #    优点：1. 批处理通常的好处：提高吞吐量；2. 减少梯度波动
        #    缺点：1. 需要更多显存；2. 过拟合风险增加，泛化能力下降
        batch_size=batch_size,
        # 主进程将采样数据平均分配给多少个 worker 并发处理
        # worker 通过 ds[i] 获取其负责的 Simples
        # 最终提交给主进程合并成输入张量（[batch_size, seq_len]）
        # 并存储到 CPU 内存
        num_workers=4,
        # 是否 "直接" 分配到锁页内存
        # linux pinning memory：不会被换页，即物理内存地址不变，便于后续 GPU 通过 DMA 复制到显存
        # Mac 运行时警告是正常的。因为其是共享内存架构，不需要 “搬运”
        pin_memory=True,
        # 预加载因子，4表示预加载4个batch的数据
        prefetch_factor=4,
    )


def _tokenize(
    samples: transformers.tokenization_utils_base.BatchEncoding,
) -> transformers.tokenization_utils_base.BatchEncoding:
    """
                ["你好，世界", "这是一个测试"]
                        |
                        V
    +-------------------------------------------+
    | 1. 分词 (Tokenization)                     |
    |  - "你好，世界" -> ["你", "好", "，", "世", "界", "[EOS]"] |
    |  - "这是一个测试" -> ["这", "是", "一", "个", "测", "试", "[EOS]"]|
    +-------------------------------------------+
                        |
                        V
    +-------------------------------------------+
    | 2. 转换为 ID (Convert to IDs)              |
    |  - ["你", "好", ...] -> [872, 1962, ..., 1] |
    |  - ["这", "是", ...] -> [6821, 3221, ..., 1] |
    +-------------------------------------------+
                        |
                        V
    +-------------------------------------------+
    | 3. 长度对齐 (Padding & Truncation)         |
    |  - [872, 1962, ...] (长度 6) -> [872, 1962, ..., 1, 0, 0] |
    |  - [6821, 3221, ...] (长度 7) -> [6821, 3221, ..., 1, 0] |
    +-------------------------------------------+
                        |
                        V
    +-------------------------------------------+
    | 4. 生成 Attention Mask & 其他              |
    |  - input_ids: [872, 1962, ..., 1, 0, 0] |
    |  - attention_mask: [1, 1, 1, ..., 1, 0, 0] |
    +-------------------------------------------+
                        |
                        V
    {"input_ids": list[list[int]], "attention_mask": list[list[int]]}

    关于 attention_mask:
    在计算注意力分数时，attention_mask==0 表示该位置是填充token，其权重为 0
    """

    texts: list[str] = samples["text"]

    """
    思考：固定 Padding vs 动态 Padding
    当前实现使用 padding="max_length"，每个样本都会被填充到 SEQ_LEN（256）。
    但实际上，一个 batch 内的样本长度可能差异很大：
    - Sample 1: "你好" -> 实际长度 3，填充到 256
    - Sample 2: "这是一个很长的文本..." -> 实际长度 200，填充到 256

    问题：大量的 [PAD] token 会带来什么影响？
    1. 计算浪费：模型需要处理大量无意义的 padding token
    2. 显存浪费：存储了很多 0

    优化思路：能否只 padding 到当前 batch 内的最大长度，而不是全局的 SEQ_LEN？
    提示：padding="longest" + DataCollator
    进阶优化：数据集打包（Packing）- 将多个短文本拼接成一个长序列，减少 padding
    """
    return model.tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=texts,
        # 固定长度。太短会填充，太长会截断
        max_length=SEQ_LEN,  # 通常小于我们所理解的上下文窗口
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )  # 没使用更简洁常见的 tokenizer(texts)，避免其他语言开发者陷入语法糖的魔法里


def _append_target_labels(
    samples: transformers.tokenization_utils_base.BatchEncoding,
) -> transformers.tokenization_utils_base.BatchEncoding:
    """
    append target labels to samples

    每个位置的标签是下一个位置的 token ID
    例如：
    input_ids: [你, 好, 世, 界, [EOS], [PAD], [PAD]]
    labels:    [好, 世, 界, [EOS], LOSS_IGNORE_LABEL, LOSS_IGNORE_LABEL, LOSS_IGNORE_LABEL]
    """
    # 转换为 Tensor 以利用向量化计算
    # 注意：torch.tensor() 会拷贝内存，直接使用 input_ids 作为 labels
    labels = torch.tensor(samples["input_ids"])  # [batch_size, seq_len]
    attention_mask = torch.tensor(samples["attention_mask"])  # [batch_size, seq_len]

    # 步骤1：把 padding 位置填充为 LOSS_IGNORE_LABEL（向量化操作）
    #        因为不希望模型学习预测 [PAD]
    labels[attention_mask == 0] = LOSS_IGNORE_LABEL

    # 步骤2：左移一位（每个位置的标签 = 下一个位置的 token）
    #        原始: [你, 好, 世, 界, [EOS], LOSS_IGNORE_LABEL, LOSS_IGNORE_LABEL]
    #        左移: [好, 世, 界, [EOS], LOSS_IGNORE_LABEL, LOSS_IGNORE_LABEL, LOSS_IGNORE_LABEL]
    labels[:, :-1] = labels[:, 1:]  # 把后面的值复制到前面
    labels[:, -1] = LOSS_IGNORE_LABEL  # 最后一位设为 LOSS_IGNORE_LABEL

    samples["labels"] = labels.tolist()
    return samples
