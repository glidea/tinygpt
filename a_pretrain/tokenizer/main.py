import datasets
import tokenizers
import tokenizers.models
import tokenizers.trainers
import tokenizers.pre_tokenizers
import tokenizers.processors
import tokenizers.decoders
import transformers


DATASET = "roneneldan/TinyStories"
VOCAB_SIZE = 2048
PAD_TOKEN = "[PAD]"
EOS_TOKEN = "[EOS]"
OUTPUT_DIR = "a_pretrain/model"


def load_dataset_texts() -> list[str]:
    raw_dataset: datasets.Dataset = datasets.load_dataset(
        path=DATASET,
        split="train",
    )

    texts: list[str] = []
    for sample in raw_dataset:
        texts.append(sample["text"])

    return texts


def new_base_tokenizer() -> tokenizers.Tokenizer:
    tokenizer = tokenizers.Tokenizer(
        model=tokenizers.models.BPE(),
    )

    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(
        add_prefix_space=False
    )

    tokenizer.decoder = tokenizers.decoders.ByteLevel()

    return tokenizer


def train() -> transformers.PreTrainedTokenizerFast:
    """
    训练 BPE Tokenizer

    BPE 训练流程：
    1. 初始化：将文本分解为字节 (Byte-Level)
    2. 统计：统计相邻字节对的出现频率
    3. 合并：不断合并高频字节对，直到词表达到目标大小
    """

    print(f"{'='*60}\n正在加载数据集: {DATASET}...\n{'='*60}")
    texts: list[str] = load_dataset_texts()

    print(f"{'='*60}\n正在训练 BPE Tokenizer...\n{'='*60}")
    tokenizer: tokenizers.Tokenizer = new_base_tokenizer()
    tokenizer.train_from_iterator(
        iterator=texts,
        trainer=tokenizers.trainers.BpeTrainer(
            vocab_size=VOCAB_SIZE,
            special_tokens=[PAD_TOKEN, EOS_TOKEN],
            show_progress=True,
        ),
    )
    tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
        single="$A " + EOS_TOKEN,  # 末尾添加 EOS
        special_tokens=[(EOS_TOKEN, tokenizer.token_to_id(EOS_TOKEN))],
    )

    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token=PAD_TOKEN,
        eos_token=EOS_TOKEN,
    )


def test(tokenizer: transformers.PreTrainedTokenizerFast):
    test_texts = [
        "I Love You",
        "Hello World",
        "123456",
    ]

    for text in test_texts:
        # Encode
        ids: list[int] = tokenizer.encode(text)
        tokens: list[str] = tokenizer.convert_ids_to_tokens(ids)

        print(f"\nOriginal: {text}")
        print(f"Token IDs: {ids}")
        print(f"Tokens: {tokens}")

        # Decode
        decoded: str = tokenizer.decode(ids, skip_special_tokens=True)
        print(f"Decoded: {decoded}")

    # 词表信息
    print("\n" + "=" * 60)
    print(f"词表大小: {tokenizer.vocab_size}")
    print("特殊 Token:")
    print(f"  - {PAD_TOKEN}: {tokenizer.pad_token_id}")
    print(f"  - {EOS_TOKEN}: {tokenizer.eos_token_id}")


if __name__ == "__main__":
    print(f"{'='*60}\n开始训练 Tokenizer...\n{'='*60}")
    tokenizer: transformers.PreTrainedTokenizerFast = train()

    print(f"{'='*60}\n测试 Tokenizer...\n{'='*60}")
    test(tokenizer)

    print(f"{'='*60}\n保存 Tokenizer 到: {OUTPUT_DIR}/\n{'='*60}")
    tokenizer.save_pretrained(OUTPUT_DIR)
