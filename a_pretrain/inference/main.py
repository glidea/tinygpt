import sys
import torch
from a_pretrain.model import model
from a_pretrain.inference import generation

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
MODEL_FILE = "a_pretrain/model/tinygpt.pt"


def main():
    if len(sys.argv) < 2:
        print(
            "用法: poetry run python -m a_pretrain.inference.main <prompt> [model_file]"
        )
        sys.exit(1)
    prompt = sys.argv[1]
    model_file = sys.argv[2] if len(sys.argv) > 2 else MODEL_FILE

    # 1. 加载模型
    print(f"正在从 {model_file} 加载模型...")
    gpt = model.GPT.load(model_file, device=DEVICE)
    print(f"模型已加载到 {DEVICE}")

    # 2. 生成文本
    print(f"\n输入: {prompt}")
    g = generation.Generator(gpt, device=DEVICE)
    result: str = g.generate(
        prompt=prompt,
        temperature=0.5,
        top_k=10,
    )
    print(f"输出: {result}")


if __name__ == "__main__":
    main()
