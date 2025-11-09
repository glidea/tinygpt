import torch
import torch.utils.data
from a_pretrain import dataset
from a_pretrain.model import model
from a_pretrain import train

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
MODEL_FILE = "a_pretrain/model/tinygpt.pt"

if __name__ == "__main__":
    print(
        f"{'='*60}\n设置随机种子。确保多次运行初始化出来的模型参数一致（训练起点一致）\n{'='*60}"
    )
    torch.manual_seed(12345)

    print(f"{'='*60}\n正在初始化模型...\n{'='*60}")
    gpt = model.GPT().to(DEVICE)

    print(f"{'='*60}\n准备训练集...\n{'='*60}")
    train_loader: torch.utils.data.DataLoader = dataset.train_stream()

    print(f"{'='*60}\n准备验证集...\n{'='*60}")
    val_loader: torch.utils.data.DataLoader = dataset.validation_stream()

    print(f"{'='*60}\n开始训练...\n{'='*60}")
    train.train(
        gpt=gpt,
        train_dataset=train_loader,
        val_dataset=val_loader,
        device=DEVICE,
        save_to=MODEL_FILE,
    )
