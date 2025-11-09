import torch
import torch.utils.data
from a_pretrain.model import model
import math
import datetime
from a_pretrain import dataset
import transformers
import time
import os

"""
思考：Epoch 数量的选择

过大的影响：
1. 过拟合：模型开始记忆训练数据而非学习泛化规律
2. 训练时间浪费：损失不再下降，继续训练无意义
3. 资源浪费：计算成本和时间成本增加

过小的影响：
1. 欠拟合：模型未充分学习数据中的模式
2. 性能不佳：模型能力未充分发挥

最佳实践：
1. 预训练大模型：通常以 Token 数量而非 Epoch 为单位
   - GPT-3: ~300B tokens
   - LLaMA: ~1T-1.4T tokens
   - 大数据集往往只需 1 个 Epoch 或更少
2. 小模型/小数据集：可能需要 3-10 个 Epoch
3. 微调（Fine-tuning）：通常 3-5 个 Epoch
4. 使用验证集监控：当验证损失不再下降时提前停止（Early Stopping）
5. 观察训练曲线：训练损失持续下降但验证损失上升 = 过拟合信号
"""
EPOCHS = 1

"""
思考：学习率（Learning Rate）的选择

定义：控制参数更新的步长。新参数 = 旧参数 - 学习率 × 梯度

异常症状：
- 过大：Loss 不稳定
- 过小：Loss 收敛极慢、易卡在局部最优

最佳实践：
- 结合 Warmup：训练初期从 0 逐步增加到最大学习率，避免初期的不稳定
- 使用学习率衰减：训练后期降低学习率，精细调优参数
- 模型越大，学习率应该越小（参数多，每个参数的微小变化都会被放大）
- Batch Size 越大，学习率可相应增大
"""
MAX_LEARNING_RATE = 7e-4
MIN_LEARNING_RATE = MAX_LEARNING_RATE * 0.1

"""
思考：Warmup 比例的选择
- 小模型/小数据集：通常 2-5% 的总步数
- 大模型/大数据集：可能需要 5-10% 甚至更长
- 经验值：0.05（5%）是一个常见且有效的选择

注意事项：
- Warmup 不宜太长，否则会浪费训练时间
- Warmup 不宜太短，否则可能无法稳定训练初期的波动
"""
WARMUP_RATIO = 0.05  # Warmup 占总训练步数的 5%

LOSS_FUNC = torch.nn.CrossEntropyLoss(
    ignore_index=dataset.LOSS_IGNORE_LABEL,
    reduction="mean",  # 对于多个样本的损失，求平均值
)

"""
思考：什么是梯度累积（Gradient Accumulation）？为什么需要它？

在训练大模型时，GPU 显存往往是瓶颈。我们可能希望使用 batch_size=64，
但显存只能容纳 batch_size=8。梯度累积让我们可以：
1. 用小 batch 做多次前向和反向传播
2. 累积梯度而不立即更新参数
3. 累积足够的梯度后再更新参数

这样就等效于使用了更大的 batch_size，而不会爆显存。

例如：ACCMULATION_STEPS=8 意味着每 8 个小 batch 更新一次参数，
等效 batch_size = 实际batch_size × 8

注意：使用梯度累积时，学习率可能需要相应增大
"""
ACCMULATION_STEPS = 1

"""
思考：为什么需要梯度裁剪（Gradient Clipping）？

在训练深度神经网络时，可能会遇到"梯度爆炸"问题：
某些参数的梯度突然变得非常大，导致参数更新幅度过大，破坏已学到的知识。

梯度裁剪通过限制梯度的最大范数（可以理解为梯度向量的长度）来解决这个问题。
如果梯度的总范数超过阈值，就按比例缩小所有梯度。

MAX_GRAD_CLIP=1.0 是一个常见的选择，但不同模型可能需要不同的值
太小可能导致学习过慢，太大可能无法有效防止梯度爆炸
"""
MAX_GRAD_CLIP = 1.0

LOG_INTERVAL = 100  # 每 100 个 step 打印一次日志
CHECKPOINT_INTERVAL = 10000  # 每 10000 个 step 保存一次临时模型


def train(
    gpt: model.GPT,
    train_dataset: torch.utils.data.DataLoader,
    val_dataset: torch.utils.data.DataLoader,
    device: str,
    save_to: str,
):
    gpt.train()  # 确保模型处于训练模式

    total_steps = EPOCHS * len(train_dataset)
    warmup_steps = int(WARMUP_RATIO * total_steps)
    print(
        f"{'='*60}\n开始训练: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print(
        f"总 Epochs: {EPOCHS}, 每个 Epoch 步数: {len(train_dataset)}, 总步数: {total_steps}"
    )
    print(f"设备: {device}, 累积步数: {ACCMULATION_STEPS}")
    print(f"学习率: 最大={MAX_LEARNING_RATE:.2e}, 最小={MAX_LEARNING_RATE * 0.1:.2e}")
    print(f"Warmup: {warmup_steps} 步 ({WARMUP_RATIO*100:.1f}% 的总步数)")
    print(f"验证集大小: {len(val_dataset)} batches")

    best_val_loss = float("inf")
    optimizer = torch.optim.AdamW(gpt.parameters())  # 优化器。负责在反向传播后更新参数
    for i in range(EPOCHS):
        print(
            f"""\n{'='*60}\nEpoch {i + 1}/{EPOCHS} 开始。
              参考：Mac mini M4 16GB 5小时一轮，
              但通常训练一两个小时就能学会语法结构，
              可通过临时 checkpoint 提前验证"""
        )
        start_time = time.time()

        train_loss: float = train_per_epoch(
            epoch_seq=i,
            total_steps=total_steps,
            gpt=gpt,
            dataset=train_dataset,
            optimizer=optimizer,
            device=device,
            tmp_checkpoint=save_to + ".tmp",
        )

        # 每个 epoch 结束后进行验证
        print(f"\n{'='*60}\nEpoch {i + 1}/{EPOCHS} 验证开始\n{'='*60}\n")
        val_loss: float = validate_per_epoch(
            gpt=gpt, val_dataset=val_dataset, device=device
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"✨ 新的最佳验证损失: {best_val_loss:.4f} 保存模型到: {save_to}")
            gpt.save(save_to)

        # 打印日志
        print(
            f"\n{'='*60}\nEpoch {i + 1} 完成，耗时: {(time.time() - start_time) / 60:.2f}分钟"
        )
        print(
            f"训练集平均 Loss: {train_loss:.4f}, 困惑度(PPL): {math.exp(train_loss):.2f}"
        )
        print(f"验证集平均 Loss: {val_loss:.4f}, 困惑度(PPL): {math.exp(val_loss):.2f}")


def train_per_epoch(
    epoch_seq: int,
    total_steps: int,
    gpt: model.GPT,
    dataset: torch.utils.data.DataLoader,
    optimizer: torch.optim.AdamW,
    device: str,
    tmp_checkpoint: str,
) -> float:
    total_loss = 0.0
    last_grad_norm_before_clip = 0.0
    last_log_time = time.time()

    for step, batch in enumerate(dataset):
        # 计算当前步数的学习率
        current_step = epoch_seq * len(dataset) + step
        lr: float = learning_rate(current_step, total_steps)
        update_optimizer_learning_rate(optimizer, lr)

        # 前向传播并计算损失
        loss: torch.Tensor = forward_and_calculate_loss(gpt, batch, device)
        total_loss += loss.item()

        # 反向传播
        (  # 基于链式法则缩放梯度。相当于多添加一个乘以 1/ACCMULATION_STEPS 的神经元
            loss / ACCMULATION_STEPS
        ).backward()

        # 累积梯度并更新参数
        if (step + 1) % ACCMULATION_STEPS == 0:
            last_grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(
                gpt.parameters(), MAX_GRAD_CLIP
            )  # 裁剪梯度。按比例缩放

            # 更新参数
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # 每 CHECKPOINT_INTERVAL 步保存一次临时模型
        if (current_step + 1) % CHECKPOINT_INTERVAL == 0:
            print(
                f"""保存临时检查点到: {tmp_checkpoint} (步数: {current_step + 1})
                  一般 loss 降到 2.5 以下说明已学习到基本语法结构，可以通过 
                  poetry run python -m a_pretrain.inference.main <prompt> a_pretrain/model/tinygpt.pt.tmp 
                  提前验证""",
                end="\n\n",
            )
            gpt.save(tmp_checkpoint)

        # 打印日志
        if (step + 1) % LOG_INTERVAL == 0:
            steps_per_sec = LOG_INTERVAL / (time.time() - last_log_time)
            remaining_steps = len(dataset) - (step + 1)

            print(
                f"[Epoch] {epoch_seq + 1}/{EPOCHS} "
                f"[Step] {step + 1}/{len(dataset)} "
                f"[学习率] {lr:.2e} "
                f"[梯度范数] {last_grad_norm_before_clip:.4f} "
                f"[Loss] {loss.item():.4f} "
                f"[困惑度] {math.exp(loss.item()):.2f} "
                f"[训练速度] {LOG_INTERVAL / (time.time() - last_log_time):.2f} steps/s "
                f"[剩余时间] {remaining_steps / steps_per_sec / 60:.2f}m "
            )
            last_log_time = time.time()

    return total_loss / len(dataset)


def validate_per_epoch(
    gpt: model.GPT, val_dataset: torch.utils.data.DataLoader, device: str
) -> float:
    gpt.eval()  # 切换到评估模式（禁用 dropout 等）
    total_loss = 0.0
    total_steps = 0

    with torch.no_grad():
        for _, batch in enumerate(val_dataset):
            loss = forward_and_calculate_loss(gpt, batch, device)
            total_loss += loss.item()
            total_steps += 1

    gpt.train()  # 切换回训练模式
    return total_loss / total_steps


def forward_and_calculate_loss(
    gpt: model.GPT,
    batch: transformers.tokenization_utils_base.BatchEncoding,
    device: str,
) -> torch.Tensor:
    # 移动数据到 GPU
    input_ids: torch.Tensor = batch["input_ids"].to(device)  # [batch_size, seq_len]
    labels: torch.Tensor = batch["labels"].to(device)  # [batch_size, seq_len]
    padding_mask: torch.Tensor = (
        batch["attention_mask"].to(device) == 0
    ).bool()  # [batch_size, seq_len]

    # 前向传播
    logits: torch.Tensor  # [batch_size, seq_len, vocab_size]
    logits, _ = gpt.__call__(
        input_ids=input_ids,
        padding_mask=padding_mask,
        past_kvs=None,  # 训练不需要
    )

    # 计算损失并反向传播（责任分配）
    loss: torch.Tensor = LOSS_FUNC(  # 标值。多个样本的 loss 值的聚合
        logits.view(
            -1, logits.size(-1)
        ),  # input. [batch_size*seq_len (样本数), vocab_size(类别数, 每个类别对应的得分)]
        labels.view(-1),  # target. [batch_size*seq_len] 每个样本的标签值
    )

    return loss


def update_optimizer_learning_rate(optimizer: torch.optim.AdamW, learning_rate: float):
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


def learning_rate(current_step: int, total_steps: int) -> float:
    """
    计算当前步数的学习率，使用 Warmup + Cosine Annealing 策略

    学习率调度分为两个阶段：
    1. Warmup 阶段（前 WARMUP_RATIO 的步数）：
       学习率从 0 线性增长到 MAX_LEARNING_RATE
    2. Cosine Annealing 阶段（剩余步数）：
       学习率按余弦曲线从 MAX_LEARNING_RATE 平滑衰减到其 0.1 倍
       - 训练中期快速下降，加速收敛
       - 训练后期缓慢下降，给模型更多时间精细调整
       - 曲线形状符合"快速接近 → 精细调整"的直觉
    """
    warmup_steps = int(WARMUP_RATIO * total_steps)

    if current_step < warmup_steps:
        # Warmup 阶段：从 0 线性增长到 MAX_LEARNING_RATE
        ratio = (current_step + 1) / warmup_steps  # 避免第0步学习率为0
        return MAX_LEARNING_RATE * ratio

    # Cosine Annealing 阶段：从 MAX_LEARNING_RATE 衰减到 MIN_LEARNING_RATE
    progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
    return (MAX_LEARNING_RATE - MIN_LEARNING_RATE) * (  # 衰减幅度，[0, delta_lr]
        0.5 * (1 + math.cos(math.pi * progress))
    ) + MIN_LEARNING_RATE  # cos(0)=1, cos(π)=-1 --> 取值范围为[0, 1]  # 加上最小学习率 --> 取值范围为[min_lr, max_lr]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 训练常见问题及优化方案
#
# 1. 过拟合 (Overfitting)
#    表现：
#    - 训练损失持续下降，但验证损失开始上升或停止下降
#    - 训练集准确率很高，但验证集/测试集表现差
#
#    优化方案：
#    - 增加训练数据量
#    - 使用更多样化的数据源
#    - 添加 Dropout
#    - 减少训练 Epochs
#    - 减少模型参数量
#
# 2. 欠拟合 (Underfitting)
#    表现：
#    - 训练损失和验证损失都很高
#    - 两者都没有明显下降趋势
#
#    优化方案：
#    - 增加模型容量（通常层数优先）
#    - 考虑减少词汇表数量，但编码效率会降低
#    - 提高数据质量（垃圾数据自身都没规律）
#    - 增加训练 Epochs
#
# 3. 梯度爆炸
#    表现：
#    - 梯度范数很大
#    - 参数更新幅度过大，Loss 剧烈波动
#
#    优化方案：
#    - 梯度裁剪
#    - 降低学习率
#    - 归一化
#
# 4. 梯度消失
#    表现：
#    - 梯度范数非常小（但不代表没有神经元“坏死”（梯度太小暂时无法学习））
#    - Loss 下降缓慢
#
#    优化方案：
#    - 残差连接
#    - 归一化
#    - 提高学习率
#    - 减少网络深度
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 性能优化方案
#
# 1. 混合精度训练
# 2. 分布式训练
