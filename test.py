# =============================================================================
# test.py — 模型测试与评估脚本
# 作用: 加载训练好的 ResNet-34 权重,在验证集/测试集上评估最终性能
#
# 与 train.py 的核心区别:
#   1. 不进行反向传播和参数更新 (torch.no_grad())
#   2. 不使用 AMP GradScaler (无梯度,不需要缩放)
#   3. 只加载模型权重 (model.state_dict),不需要恢复优化器状态
#   4. 可以进行更多详细分析: 混淆矩阵、逐类别准确率等
#
# 运行方式:
#   python test.py                                    # 使用默认配置和 best_model.pth
#   python test.py --config configs/default.yaml      # 指定配置文件
#   python test.py --checkpoint checkpoints/latest.pth  # 指定 checkpoint 路径
#   python test.py --topk 1 5 10                      # 同时计算 Top-1/5/10 准确率
# =============================================================================

# argparse: Python 标准库,用于解析命令行参数
# 让脚本可以通过 python test.py --config xxx.yaml 的方式灵活配置
import argparse

# os: 操作系统接口,用于路径操作、目录创建
import os

# time: 时间测量,用于计算推理吞吐量 (images/second)
import time

# torch: PyTorch 深度学习核心库
import torch

# torch.nn: 神经网络模块库 (用于 CrossEntropyLoss)
import torch.nn as nn

# autocast: AMP 自动混合精度上下文管理器
# 测试阶段也使用 autocast 加速前向传播 (虽然不需要梯度,但 FP16 计算更快)
from torch.cuda.amp import autocast

# ── 项目内部模块 ────────────────────────────────────────────────────────────────
# build_resnet34: 模型工厂函数,根据 num_classes 创建 ResNet-34 实例
from model import build_resnet34

# build_dataloaders: 构建高性能数据加载器 (pin_memory、多进程等)
# load_config: 从 YAML 文件读取所有配置参数
from data_processing import build_dataloaders, load_config

# get_logger: 创建同时写控制台和文件的日志记录器
from utils.logger import get_logger

# AverageMeter: 跨 batch 累积计算均值的计数器
# accuracy:     计算 Top-K 准确率
from utils.metrics import AverageMeter, accuracy


# =============================================================================
# 第一部分: 模型权重加载函数
# =============================================================================


def load_model_weights(model: nn.Module, weight_path: str, device: torch.device) -> nn.Module:
    """
    从磁盘加载模型权重到模型对象。

    支持两种格式的权重文件:
        1. 只含 state_dict 的文件 (由 save_checkpoint 的 best_model.pth 生成)
           即: torch.save(model.state_dict(), path)
           加载: model.load_state_dict(torch.load(path))

        2. 完整 Checkpoint 文件 (由 save_checkpoint 的 latest.pth 生成)
           即: torch.save({"model_state_dict": ..., "epoch": ..., ...}, path)
           加载: model.load_state_dict(checkpoint["model_state_dict"])

    函数会自动判断文件类型,无需手动区分。

    参数:
        model       (nn.Module):    已初始化的模型对象 (结构必须与权重匹配)
        weight_path (str):          权重文件路径
        device      (torch.device): 目标设备,权重 Tensor 会加载到此设备

    返回:
        nn.Module: 加载了权重的模型对象 (与输入的 model 是同一对象,in-place 修改)
    """

    # ── 检查文件是否存在 ──────────────────────────────────────────────────────
    # 提前检查并给出明确错误信息,而不是让 PyTorch 抛出难以理解的异常
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(
            f"权重文件不存在: {weight_path}\n"
            f"请先运行 train.py 训练模型,或检查 --checkpoint 参数是否正确。"
        )

    # ── 加载权重文件到内存 ────────────────────────────────────────────────────
    # torch.load: 反序列化 .pth/.pt 文件
    # map_location=device: 将权重 Tensor 直接加载到目标设备,避免先加载到 CPU 再转移
    checkpoint_data = torch.load(weight_path, map_location=device)

    # ── 自动判断文件格式并提取 state_dict ─────────────────────────────────────
    # isinstance(obj, dict): 判断对象是否是字典类型
    # 完整 Checkpoint 是 dict 且包含 "model_state_dict" key
    # 纯 state_dict 也是 dict,但 key 是层的名称 (如 "stem_conv.weight")
    if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
        # 格式一: 完整 Checkpoint — 提取 model_state_dict 子字典
        state_dict = checkpoint_data["model_state_dict"]

        # 打印额外信息,方便用户了解这个 checkpoint 的来源
        saved_epoch = checkpoint_data.get("epoch", "未知")
        saved_top1 = checkpoint_data.get("best_top1", checkpoint_data.get("val_top1", 0.0))
        print(f"[加载权重] 完整 Checkpoint 格式 | 保存于 Epoch {saved_epoch} | Top-1: {saved_top1:.2f}%")
    else:
        # 格式二: 纯 state_dict — 直接使用
        state_dict = checkpoint_data
        print(f"[加载权重] 纯 state_dict 格式")

    # ── 将 state_dict 写入模型 ────────────────────────────────────────────────
    # load_state_dict: 逐一将 state_dict 中的参数值写入模型对应层
    # strict=True (默认): 要求完全匹配 — key 不能多也不能少
    # strict=False: 允许部分匹配 — 用于迁移学习或修改了部分层的情况
    model.load_state_dict(state_dict, strict=True)

    print(f"[加载权重] 成功从 {weight_path} 加载权重")
    return model


# =============================================================================
# 第二部分: 评估函数
# =============================================================================


@torch.no_grad()  # 装饰器: 整个函数内不计算梯度,节省显存和时间
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    topk: tuple,
    logger,
) -> dict:
    """
    在数据集上评估模型性能,返回各项指标。

    评估流程:
        1. 设置 model.eval() → BN 使用训练时统计的全局均值/方差
        2. 遍历所有 batch → 前向传播 → 计算 loss 和 Top-K 准确率
        3. 汇总所有 batch 的指标 → 计算全局均值

    参数:
        model     (nn.Module):    待评估的模型 (已加载权重)
        loader    (DataLoader):   数据加载器 (通常是验证集或测试集)
        criterion (nn.Module):    损失函数 (CrossEntropyLoss)
        device    (torch.device): 计算设备
        topk      (tuple):        要计算的 Top-K 值,如 (1, 5)
        logger:                   日志记录器

    返回:
        dict: 包含以下指标的字典:
              - "loss":   float,整个数据集的平均损失
              - "top1":   float,Top-1 准确率 (%)
              - "topk":   dict, 所有 K 值的准确率,如 {"top1": 75.2, "top5": 92.3}
              - "throughput": float,推理吞吐量 (images/second)
              - "total_samples": int,评估的总样本数
    """

    # ── 切换到评估模式 ────────────────────────────────────────────────────────
    # model.eval() 的效果:
    #   1. BatchNorm 层: 使用训练时统计的 running_mean 和 running_var
    #      而非当前 batch 的统计量 (保证推理结果不依赖 batch 大小)
    #   2. Dropout 层: 所有神经元全部激活 (p=0),不随机丢弃
    model.eval()

    # ── 初始化指标计数器 ──────────────────────────────────────────────────────
    # 每个 Top-K 值创建一个独立的 AverageMeter
    loss_meter = AverageMeter("Loss")

    # topk_meters: 字典,key=K值, value=对应的 AverageMeter
    # 例如: {1: AverageMeter("Top-1"), 5: AverageMeter("Top-5")}
    topk_meters = {k: AverageMeter(f"Top-{k}") for k in topk}

    # ── 记录推理开始时间 ──────────────────────────────────────────────────────
    # time.perf_counter(): 高精度计时器,精度到纳秒级别
    # 用于计算整个评估阶段的吞吐量
    eval_start = time.perf_counter()

    # ── 主评估循环 ────────────────────────────────────────────────────────────
    # 注意: 这里使用了 @torch.no_grad() 装饰器,整个函数内梯度计算被禁用
    # 效果: 不建立计算图 → 节省约 50% 显存 → 允许更大的 batch size
    for batch_idx, (images, labels) in enumerate(loader):

        # ── 将数据搬运到计算设备 ──────────────────────────────────────────────
        # non_blocking=True: 异步传输 (配合 pin_memory=True 使用)
        # CPU 提交传输请求后立即返回继续处理下一条指令,GPU 通过 DMA 读取数据
        images = images.to(device, non_blocking=True)  # [N, 3, H, W] → GPU
        labels = labels.to(device, non_blocking=True)  # [N] → GPU

        # ── 前向传播 (FP16 加速) ──────────────────────────────────────────────
        # autocast() 在 FP16 精度下执行矩阵运算,加快推理速度
        # 即使测试阶段不需要梯度,FP16 前向传播仍比 FP32 快约 2x
        with autocast():
            # 前向传播: [N, 3, 224, 224] → [N, num_classes]
            logits = model(images)

            # 计算损失: 衡量模型预测与真实标签的差距
            loss = criterion(logits, labels)

        # ── 计算 Top-K 准确率 ─────────────────────────────────────────────────
        # accuracy 返回一个列表,长度等于 topk 的长度
        # 例如 topk=(1,5) → acc_values = [top1_acc, top5_acc]
        acc_values = accuracy(logits, labels, topk=topk)

        # ── 更新各指标计数器 ──────────────────────────────────────────────────
        # images.size(0): 当前 batch 的实际样本数 (最后一个 batch 可能不完整)
        batch_n = images.size(0)

        # 更新 loss 计数器 (加权均值)
        # loss.item(): 将 Tensor 标量转换为 Python 浮点数
        loss_meter.update(loss.item(), batch_n)

        # 更新每个 Top-K 的计数器
        # zip(topk, acc_values) 将 K 值和对应的准确率配对迭代
        for k, acc_val in zip(topk, acc_values):
            topk_meters[k].update(acc_val, batch_n)

        # ── 每 50 个 batch 打印一次进度 ───────────────────────────────────────
        if (batch_idx + 1) % 50 == 0:
            logger.info(
                f"  [{batch_idx+1}/{len(loader)}] "
                f"Loss: {loss_meter.avg:.4f}  "
                + "  ".join(
                    f"Top-{k}: {topk_meters[k].avg:.2f}%" for k in topk
                )
            )

    # ── 计算整体吞吐量 ────────────────────────────────────────────────────────
    eval_elapsed = time.perf_counter() - eval_start   # 总耗时 (秒)
    total_samples = len(loader.dataset)                # 数据集中的总样本数
    throughput = total_samples / eval_elapsed          # 每秒处理的图片数

    # ── 构建并返回结果字典 ────────────────────────────────────────────────────
    results = {
        # 平均损失
        "loss": loss_meter.avg,

        # Top-1 准确率 (最常用的指标,单独提出方便访问)
        "top1": topk_meters[1].avg if 1 in topk_meters else None,

        # 所有 Top-K 的准确率字典: {1: 75.2, 5: 92.3, ...}
        "topk": {k: topk_meters[k].avg for k in topk},

        # 推理吞吐量
        "throughput": throughput,

        # 评估的总样本数
        "total_samples": total_samples,

        # 总耗时 (秒)
        "elapsed": eval_elapsed,
    }

    return results


# =============================================================================
# 第三部分: 主函数
# =============================================================================


def main(config_path: str, checkpoint_path: str, topk: tuple):
    """
    测试主函数: 加载模型 → 评估 → 打印结果

    参数:
        config_path     (str):   配置文件路径
        checkpoint_path (str):   权重文件路径 (None 时使用默认路径)
        topk            (tuple): 要计算的 Top-K 值列表
    """

    # ── 加载配置文件 ──────────────────────────────────────────────────────────
    # load_config 读取 YAML 配置文件,返回嵌套字典
    cfg = load_config(config_path)

    # ── 初始化日志记录器 ──────────────────────────────────────────────────────
    # get_logger 创建一个同时输出到控制台和文件的 Logger
    log_dir = cfg.get("log_dir", "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = get_logger(
        name="test",
        log_file=os.path.join(log_dir, "test.log")
    )

    logger.info("=" * 70)
    logger.info("ResNet-34 模型评估")
    logger.info(f"配置文件: {config_path}")
    logger.info("=" * 70)

    # ── 确定 Checkpoint 路径 ──────────────────────────────────────────────────
    # 如果用户没有通过 --checkpoint 指定路径,使用默认的 best_model.pth
    if checkpoint_path is None:
        checkpoint_dir = cfg.get("checkpoint_dir", "checkpoints")
        checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
        logger.info(f"未指定 --checkpoint,使用默认路径: {checkpoint_path}")

    # ── 设备选择 ──────────────────────────────────────────────────────────────
    # 与 train.py 一致: 优先使用 GPU
    if not torch.cuda.is_available():
        logger.warning("⚠️  未检测到 CUDA GPU!将在 CPU 上评估 (速度较慢)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"评估设备: {device}")

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {gpu_name}  显存: {gpu_mem:.1f} GB")

        # 对于固定输入尺寸 (224×224),cuDNN benchmark 能找到最快的卷积算法
        torch.backends.cudnn.benchmark = True

    # ── 构建数据加载器 ────────────────────────────────────────────────────────
    # 测试阶段也需要数据加载器,这里我们使用 val_loader 作为测试集
    # (真实项目中可能有独立的 test split,此处用 val 代替)
    logger.info("构建数据加载器...")
    _, val_loader = build_dataloaders(cfg)
    # 仅使用 val_loader,弃用 train_loader (用 _ 表示不需要的返回值)

    # ── 构建模型并加载权重 ────────────────────────────────────────────────────
    num_classes = cfg["model"]["num_classes"]
    logger.info(f"构建 ResNet-34 模型 (类别数: {num_classes})")
    model = build_resnet34(num_classes=num_classes)

    # 加载训练好的权重
    model = load_model_weights(model, checkpoint_path, device)

    # 将模型移动到目标设备 (GPU/CPU)
    model = model.to(device)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数量: {total_params / 1e6:.2f}M")

    # ── 构建损失函数 ──────────────────────────────────────────────────────────
    # 测试时不使用 label_smoothing (测试集上的 loss 应该反映真实性能)
    criterion = nn.CrossEntropyLoss().to(device)

    # ── 确保 topk 中包含 1 (Top-1 是最基本的指标) ────────────────────────────
    # set(): 去重 | sorted(): 排序 | tuple(): 转回元组
    topk = tuple(sorted(set([1] + list(topk))))
    logger.info(f"评估指标: Top-{topk}")

    # ── 开始评估 ──────────────────────────────────────────────────────────────
    logger.info(f"\n开始在验证集上评估...")
    logger.info(f"{'─'*60}")

    # 调用评估函数
    results = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        topk=topk,
        logger=logger,
    )

    # ── 打印最终结果 ──────────────────────────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info("评估结果汇总")
    logger.info(f"{'─'*70}")
    logger.info(f"  总样本数:    {results['total_samples']:,}")
    logger.info(f"  总耗时:      {results['elapsed']:.1f}s")
    logger.info(f"  吞吐量:      {results['throughput']:.0f} img/s")
    logger.info(f"  平均 Loss:   {results['loss']:.4f}")
    logger.info(f"{'─'*70}")

    # 打印所有 Top-K 准确率
    for k, acc in results["topk"].items():
        # 根据 ResNet-34 的参考精度给出评估
        # ResNet-34 ImageNet Top-1 参考值约为 73.3%,Top-5 约为 91.4%
        if k == 1:
            reference = 73.3
            status = "✅ 达到参考精度" if acc >= reference else f"⚠️  参考精度约 {reference}%"
        elif k == 5:
            reference = 91.4
            status = "✅ 达到参考精度" if acc >= reference else f"⚠️  参考精度约 {reference}%"
        else:
            status = ""
        logger.info(f"  Top-{k} 准确率: {acc:.2f}%  {status}")

    logger.info(f"{'='*70}")

    # 返回结果字典 (方便集成测试或进一步分析)
    return results


# =============================================================================
# 命令行入口
# =============================================================================

if __name__ == "__main__":

    # ── 构建命令行参数解析器 ──────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="ResNet-34 模型评估脚本",
        # formatter_class 设置帮助文档的格式风格
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --config 参数: 指定配置文件路径
    # default: 默认值,当用户不传此参数时使用
    # help: 在 python test.py --help 时显示的说明文字
    parser.add_argument(
        "--config",
        type=str,                           # 参数类型: 字符串
        default="configs/default.yaml",     # 默认值
        help="配置文件路径",
    )

    # --checkpoint 参数: 指定权重文件路径
    # 如果不指定,程序会使用 checkpoint_dir/best_model.pth
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,                       # None = 使用默认路径
        help="模型权重文件路径 (默认使用 checkpoint_dir/best_model.pth)",
    )

    # --topk 参数: 指定要计算的 Top-K 值
    # nargs="+": 接受一个或多个值 (python test.py --topk 1 5 10)
    # type=int:  每个值解析为整数
    parser.add_argument(
        "--topk",
        nargs="+",                          # 接受多个值: --topk 1 5 10
        type=int,
        default=[1, 5],                     # 默认计算 Top-1 和 Top-5
        help="要计算的 Top-K 准确率,可传多个值 (默认: 1 5)",
    )

    # ── 解析命令行参数 ────────────────────────────────────────────────────────
    # parse_args() 从 sys.argv 读取命令行输入并解析
    # args.config, args.checkpoint, args.topk 即对应的参数值
    args = parser.parse_args()

    # ── 启动评估主函数 ────────────────────────────────────────────────────────
    # tuple(args.topk): 将列表转为元组 (accuracy 函数接受 tuple 类型)
    main(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        topk=tuple(args.topk),
    )
