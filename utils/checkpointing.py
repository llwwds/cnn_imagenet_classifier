# =============================================================================
# utils/checkpointing.py — 模型检查点保存与加载模块
# 功能:
#   1. save_checkpoint: 将训练状态完整保存到磁盘
#   2. load_checkpoint: 从磁盘恢复训练状态,支持断点续训
# =============================================================================
#
# 什么是 Checkpoint (检查点)?
#   在长时间训练中 (ImageNet 训练需要几十小时),意外中断是常见情况:
#     - 服务器断电/重启
#     - 程序崩溃
#     - 手动停止 (想调整超参数)
#   Checkpoint 是训练状态的"快照",包含恢复训练所需的全部信息。
#   有了 Checkpoint,无需从头重新训练,直接从最后一个保存点继续。
#
# 一个完整的 Checkpoint 应该包含:
#   1. model.state_dict()     — 模型权重 (最重要,参数量最大)
#   2. optimizer.state_dict() — 优化器状态 (SGD 的动量缓冲区)
#   3. scheduler.state_dict() — 学习率调度器状态 (当前 epoch 进度)
#   4. scaler.state_dict()    — AMP GradScaler 状态 (FP16 缩放因子)
#   5. epoch                  — 当前 epoch 编号 (知道从哪里继续)
#   6. best_top1              — 历史最优 Top-1 准确率 (知道最好成绩)
#
# 什么是 state_dict?
#   state_dict 是 PyTorch 中所有有状态对象 (模型、优化器等) 的序列化格式。
#   它是一个 Python OrderedDict,将参数名映射到 Tensor:
#     {"stem_conv.weight": tensor(...), "stage1.0.conv1.weight": tensor(...), ...}
#   通过 state_dict/load_state_dict 可以安全地保存和恢复任意 PyTorch 对象的状态。
# =============================================================================

# os: 文件系统操作 (路径拼接、目录创建)
import os

# torch: PyTorch 核心库,提供 torch.save/torch.load 序列化接口
import torch


# =============================================================================
# 函数一: save_checkpoint — 保存训练状态到磁盘
# =============================================================================


def save_checkpoint(
    checkpoint_dir: str,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.cuda.amp.GradScaler,
    val_top1: float,
    best_top1: float,
    is_best: bool,
) -> None:
    """
    将当前训练状态保存为 Checkpoint 文件。

    保存策略 (双文件):
        1. latest.pth:    每个 epoch 结束后都覆盖保存 → 用于断点续训
        2. best_model.pth: 只在 val_top1 达到历史最优时才保存 → 用于最终评估

    为什么保存两个文件?
        - latest.pth 保证最近的训练状态可以恢复 (即使最后几个 epoch 精度下降)
        - best_model.pth 保存泛化能力最强的权重 (测试时应使用这个文件)

    参数:
        checkpoint_dir (str):         保存目录,如 "checkpoints"
        epoch          (int):         当前 epoch 编号 (1-based)
        model          (nn.Module):   待保存的模型
        optimizer      (Optimizer):   优化器 (保存动量状态)
        scheduler:                    学习率调度器
        scaler         (GradScaler):  AMP 梯度缩放器
        val_top1       (float):       当前 epoch 的验证集 Top-1 准确率
        best_top1      (float):       历史最优验证集 Top-1 准确率
        is_best        (bool):        当前 epoch 是否是历史最优

    返回:
        None (直接写文件,无返回值)
    """

    # ── 确保保存目录存在 ───────────────────────────────────────────────────────
    # os.makedirs: 递归创建目录树,即使父目录不存在也能一次性创建
    # exist_ok=True: 如果目录已存在,不抛出异常 (默认行为会报 FileExistsError)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── 构建完整的 Checkpoint 字典 ────────────────────────────────────────────
    # 这个字典包含恢复训练所需的全部信息
    # torch.save 会将这个字典序列化 (pickle) 到磁盘
    checkpoint = {
        # epoch: 记录当前完成的 epoch 编号
        # 加载时: start_epoch = checkpoint["epoch"] + 1 (从下一个 epoch 继续)
        "epoch": epoch,

        # model_state_dict: 模型的所有可学习参数 (权重和偏置)
        # .state_dict() 返回 OrderedDict: {"layer_name": tensor, ...}
        # 注意: 如果模型用了 DataParallel,需要保存 model.module.state_dict()
        "model_state_dict": model.state_dict(),

        # optimizer_state_dict: 优化器内部状态
        # SGD 的状态包括: 每个参数的动量缓冲区 (momentum buffer)
        # 这些状态对于继续训练的精度非常重要,如果不保存,恢复后精度会下降
        "optimizer_state_dict": optimizer.state_dict(),

        # scheduler_state_dict: 学习率调度器状态
        # 包含: 当前 epoch 计数、各阶段进度等
        # 保证续训时学习率从正确的位置继续变化
        "scheduler_state_dict": scheduler.state_dict(),

        # scaler_state_dict: AMP GradScaler 状态
        # 包含: 当前 scale factor (动态调整的 FP16 梯度缩放系数)
        # 保证续训时 FP16 训练的数值稳定性
        "scaler_state_dict": scaler.state_dict(),

        # val_top1: 当前 epoch 的验证集准确率,方便查看 checkpoint 时了解训练进度
        "val_top1": val_top1,

        # best_top1: 历史最优验证集准确率,续训时恢复这个基准
        "best_top1": best_top1,
    }

    # ── 保存最新 Checkpoint (每个 epoch 覆盖) ────────────────────────────────
    # os.path.join: 跨平台路径拼接 (Windows/Linux 兼容)
    # 示例: "checkpoints" + "latest.pth" → "checkpoints/latest.pth"
    latest_path = os.path.join(checkpoint_dir, "latest.pth")

    # torch.save: 将 Python 对象序列化并保存到磁盘
    # 内部使用 pickle 协议,支持 Tensor、dict、list 等任意 Python 对象
    torch.save(checkpoint, latest_path)

    # ── 如果是历史最优,额外保存 best_model.pth ───────────────────────────────
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pth")

        # 直接保存模型权重 (state_dict) 而非完整 checkpoint
        # 原因: best_model.pth 主要用于推理/测试,不需要优化器状态
        # 体积更小 (约 85MB vs 完整 checkpoint 的约 340MB)
        torch.save(model.state_dict(), best_path)

        print(
            f"[Checkpoint] 新最优模型已保存: {best_path}  "
            f"(Top-1: {best_top1:.2f}%)"
        )

    # 打印保存日志
    print(
        f"[Checkpoint] Epoch {epoch} 已保存: {latest_path}  "
        f"(Val Top-1: {val_top1:.2f}%)"
    )


# =============================================================================
# 函数二: load_checkpoint — 从磁盘恢复训练状态
# =============================================================================


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> tuple:
    """
    从 Checkpoint 文件恢复完整的训练状态。

    参数:
        path      (str):          Checkpoint 文件路径,如 "checkpoints/latest.pth"
        model     (nn.Module):    已初始化的模型对象 (会被 in-place 修改)
        optimizer (Optimizer):    已初始化的优化器对象 (会被 in-place 修改)
        scheduler:                已初始化的调度器对象 (会被 in-place 修改)
        scaler    (GradScaler):   已初始化的 GradScaler 对象 (会被 in-place 修改)
        device    (torch.device): 目标设备 (cuda/cpu),用于指定 Tensor 加载位置

    返回:
        tuple: (start_epoch: int, best_top1: float)
               start_epoch: 下一个要训练的 epoch 编号 (即 checkpoint["epoch"] + 1)
               best_top1:   历史最优 Top-1 准确率

    使用示例:
        model = build_resnet34()
        optimizer = build_optimizer(model, cfg)
        scheduler = build_scheduler(optimizer, cfg)
        scaler = GradScaler()
        start_epoch, best_top1 = load_checkpoint(
            "checkpoints/latest.pth",
            model, optimizer, scheduler, scaler,
            device=torch.device("cuda")
        )
        # 从 start_epoch 继续训练
        for epoch in range(start_epoch, total_epochs + 1):
            ...
    """

    # ── 文件存在性检查 ────────────────────────────────────────────────────────
    # 在尝试加载之前先检查文件是否存在,给出清晰的错误提示
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Checkpoint 文件不存在: {path}\n"
            f"请确认路径正确,或检查 configs/default.yaml 中的 train.resume 设置。"
        )

    # ── 加载 Checkpoint 字典 ──────────────────────────────────────────────────
    # torch.load: 反序列化 checkpoint 文件
    # map_location=device: 将 Tensor 加载到指定设备
    #   - 如果 checkpoint 在 GPU 上保存,但当前机器无 GPU,
    #     可以用 map_location="cpu" 加载到 CPU
    #   - 如果不指定,加载到原来保存时的设备 (可能报错)
    checkpoint = torch.load(path, map_location=device)

    # ── 恢复模型权重 ──────────────────────────────────────────────────────────
    # load_state_dict: 将 state_dict 中的参数值逐一写入模型的对应层
    # strict=True (默认): 要求 state_dict 的 key 与模型结构完全匹配
    #   如果修改了模型架构 (增删了层),需要 strict=False 忽略不匹配的 key
    model.load_state_dict(checkpoint["model_state_dict"])

    # ── 恢复优化器状态 ────────────────────────────────────────────────────────
    # 恢复 SGD 的动量缓冲区,保证续训时优化动态与中断前一致
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # ── 恢复学习率调度器状态 ──────────────────────────────────────────────────
    # 恢复调度器的 epoch 计数,保证学习率从正确的值继续变化
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # ── 恢复 AMP GradScaler 状态 ──────────────────────────────────────────────
    # 恢复 FP16 的梯度缩放系数,保证数值稳定性
    scaler.load_state_dict(checkpoint["scaler_state_dict"])

    # ── 读取训练进度 ──────────────────────────────────────────────────────────
    # checkpoint["epoch"] 是最后完成的 epoch 编号
    # +1 表示从下一个 epoch 开始训练
    start_epoch = checkpoint["epoch"] + 1

    # 读取历史最优准确率,作为后续 is_best 判断的基准
    best_top1 = checkpoint.get("best_top1", 0.0)
    # .get() 提供默认值 0.0,兼容旧格式的 checkpoint 文件

    print(
        f"[Checkpoint] 已从 {path} 恢复训练状态  "
        f"(上次完成 Epoch {checkpoint['epoch']}, "
        f"历史最优 Top-1: {best_top1:.2f}%)"
    )

    return start_epoch, best_top1


# =============================================================================
# 快速验证入口
# =============================================================================

if __name__ == "__main__":
    import tempfile  # 临时目录,用于测试时避免污染真实目录

    # 导入模型用于测试
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model import build_resnet34

    print("测试 save_checkpoint 和 load_checkpoint...")

    # 1. 创建一个小模型 (10类,快速测试)
    model = build_resnet34(num_classes=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30)
    scaler = torch.cuda.amp.GradScaler()
    device = torch.device("cpu")  # 测试在 CPU 上进行

    # 2. 保存 checkpoint 到临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        save_checkpoint(
            checkpoint_dir=tmpdir,
            epoch=5,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            val_top1=65.0,
            best_top1=65.0,
            is_best=True,
        )

        # 3. 创建新的模型/优化器对象,从 checkpoint 恢复
        model2 = build_resnet34(num_classes=10)
        optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01, momentum=0.9)
        scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=30)
        scaler2 = torch.cuda.amp.GradScaler()

        start_epoch, best_top1 = load_checkpoint(
            path=os.path.join(tmpdir, "latest.pth"),
            model=model2,
            optimizer=optimizer2,
            scheduler=scheduler2,
            scaler=scaler2,
            device=device,
        )

        print(f"恢复成功: start_epoch={start_epoch}, best_top1={best_top1}%")
        assert start_epoch == 6, f"期望 start_epoch=6,实际={start_epoch}"
        assert best_top1 == 65.0, f"期望 best_top1=65.0,实际={best_top1}"

    print("所有测试通过 ✅")
