# A) 预训练

> 预训练（Pre-training）是 LLM 学习的基础。在这个阶段，模型会通过学习海量文本数据来掌握语言的语法、事实知识以及一定的推理能力。这个过程不针对任何特定任务，而是为了构建一个通用的语言理解基础。

### 模型与数据集简介

- **模型**：我们将从零开始训练一个约 10M 参数的迷你 GPT 模型。它采用了经典的 Decoder-only Transformer 架构，具体参数如下：
  - `hidden_size`: 256
  - `num_layers`: 12
  - `num_heads`: 4
  - `vocab_size`: 2048
  - `max_seq_len`: 256 (上下文窗口)

- **数据集**：使用 [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) 数据集，其中包含大量词汇简单的儿童故事。使得在本地快速训练出一个能理解基本语法并生成连贯文本的模型 ([论文](https://arxiv.org/abs/2305.07759))

---

![](/images/preview1.png)

---

### 步骤0：安装依赖

```bash
pipx install poetry && poetry install
```

### 步骤1：训练分词器

这一步会使用 `tinystories` 数据集训练一个 BPE 分词器，并将训练好的分词器保存在 `a_pretrain/model/` 目录下。

```bash
poetry run python -m a_pretrain.tokenizer.main
```

### 步骤2：训练模型

这一步会使用上一步训练好的分词器来训练一个小型 GPT 模型。模型会保存在 `a_pretrain/model/` 目录下。

```bash
poetry run python -m a_pretrain.main
```

### 步骤3：推理验证

使用训练好的模型进行文本补全。

```bash
poetry run python -m a_pretrain.inference.main "你的 Prompt"

# 比如
poetry run python -m a_pretrain.inference.main "Once upon a time"
```

> [!NOTE]
> 预训练模型只能补全文本，不能对话
> 只会英语

