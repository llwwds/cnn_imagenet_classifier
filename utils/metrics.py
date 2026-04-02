# =============================================================================
# utils/metrics.py — 训练指标工具模块
# 功能:
#   1. AverageMeter: 累积计算均值的计数器 (用于 loss、accuracy、time 等)
#   2. accuracy:     批次级别的 Top-K 准确率计算函数
# =============================================================================
#
# 为什么需要这个模块?
#
# 在深度学习训练中,我们需要跨越整个 epoch 跟踪多项指标:
#   - 每个 batch 的 loss 是不同的,需要计算整个 epoch 的平均 loss
#   - 准确率需要对所有样本计数,而不是简单平均各 batch 的准确率
#     (因为最后一个 batch 可能样本数不同,简单平均会有偏差)
#
# 什么是 Top-K 准确率?
#   Top-1: 模型预测的第1名类别与真实标签匹配 → 即通常意义的"正确"
#   Top-5: 模型预测的前5个类别中包含真实标签 → ImageNet 的标准评估指标之一
#
#   为什么 ImageNet 用 Top-5?
#   ImageNet 有 1000 个类别,其中很多类别之间视觉上非常相似
#   (例如: 200+ 种狗的品种)。Top-5 更宽松,能更好地反映模型的实际能力。
# =============================================================================

# torch: PyTorch 核心库,提供 Tensor 运算
import torch


# =============================================================================
# 类一: AverageMeter — 滑动均值计数器
# =============================================================================


class AverageMeter:
    """
    跟踪并计算一个指标的累积均值。

    设计思路:
        不需要存储所有历史值,只维护以下4个状态变量:
        - val:   最近一次 update 的值 (方便打印当前 batch 的即时值)
        - avg:   到目前为止所有 update 的加权平均值 (epoch 级别的综合指标)
        - sum:   所有 update 的加权累计和
        - count: 累计的样本总数

    加权计算原因:
        每个 batch 的样本数量可能不同 (最后一个 batch 可能较小)。
        如果对各 batch 的 loss 直接平均,会给小 batch 同等权重,结果有偏差。
        加权平均 (按样本数加权) 才是真正的整体均值。

    参数:
        name (str): 这个指标的名字,用于调试打印,如 "Loss"、"Top-1 Acc"

    使用示例:
        meter = AverageMeter("Loss")
        for images, labels in loader:
            loss = compute_loss(...)
            meter.update(loss.item(), n=images.size(0))  # n=batch_size
        print(f"Epoch 平均 Loss: {meter.avg:.4f}")
    """

    def __init__(self, name: str):
        # 存储这个指标的名称,方便调试时识别是哪个指标
        # 例如: AverageMeter("Loss"), AverageMeter("Top-1 Acc")
        self.name = name

        # 调用 reset() 初始化所有统计变量为 0
        # 这样可以复用同一个 AverageMeter 对象 (每个 epoch 开始时 reset)
        self.reset()

    def reset(self):
        """
        将所有累积统计量重置为零。

        通常在每个 epoch 开始时调用,清空上一个 epoch 的数据。
        也可以在 __init__ 中调用,避免重复写 4 行初始化代码。
        """
        # val: 最近一次 update 的值 (即当前 batch 的指标值)
        # 作用: 在训练日志中显示"当前 batch"的实时数值
        self.val = 0.0

        # avg: 加权均值 = sum / count
        # 作用: epoch 结束时的最终指标 (例如: 该 epoch 的平均 loss)
        self.avg = 0.0

        # sum: 所有 update 调用的 val × n 的累计总和
        # 例如: 3 个 batch, loss=[2.0, 1.5, 1.8], n=[1024, 1024, 576]
        #        sum = 2.0×1024 + 1.5×1024 + 1.8×576 = 5140.8
        self.sum = 0.0

        # count: 累计的样本总数 (即所有 update 的 n 之和)
        # 例如: sum(n) = 1024 + 1024 + 576 = 2624
        self.count = 0

    def update(self, val: float, n: int = 1):
        """
        用新的观测值更新统计量。

        参数:
            val (float): 当前 batch 的指标值
                         对于 loss: 通常是 loss.item() (一个 Python 浮点数)
                         对于 accuracy: 通常是百分比 (0.0~100.0)
            n   (int):   当前 batch 的样本数量 (权重)
                         通常是 images.size(0),即当前 batch 的实际大小
                         最后一个 batch 可能比 batch_size 小,因此要传入实际值
                         默认为 1 (不加权,简单均值)

        示例:
            # batch_size=1024, loss=2.3456
            meter.update(2.3456, n=1024)

            # 最后一个不完整 batch: batch_size=576, loss=1.9
            meter.update(1.9, n=576)
        """
        # 更新最近一次的值 (用于显示实时状态)
        self.val = val

        # 加权累加: 将 val × n 加到总和中
        # 这样 sum / count 才是正确的加权平均
        self.sum += val * n

        # 累计样本总数
        self.count += n

        # 重新计算加权均值
        # 防御性除法: 虽然正常使用时 count 不会为 0,但加上保护更安全
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __repr__(self) -> str:
        """
        定义 print(meter) 时的输出格式。
        __repr__ 是 Python 的特殊方法 (dunder method),
        用于定义对象的"官方"字符串表示。
        """
        return f"AverageMeter(name={self.name}, avg={self.avg:.4f}, count={self.count})"


# =============================================================================
# 函数二: accuracy — Top-K 准确率计算
# =============================================================================


def accuracy(
    output: torch.Tensor,
    target: torch.Tensor,
    topk: tuple = (1,),
) -> list:
    """
    计算一个 batch 内的 Top-K 准确率。

    算法思路:
        1. 对 logits (每个类别的原始分数) 按分数从高到低排序
        2. 取前 K 个类别的索引
        3. 检查真实标签是否在前 K 个预测中出现
        4. 统计命中的样本数量,除以总样本数,得到准确率 (百分比)

    参数:
        output (Tensor): 模型输出的 logits,形状 [N, C]
                         N = batch size, C = 类别数
                         值越大表示模型认为该类别的概率越高
        target (Tensor): 真实标签,形状 [N]
                         每个值是一个整数类别索引 (0 ~ C-1)
        topk   (tuple):  要计算的 K 值列表
                         例如 (1, 5) 表示同时计算 Top-1 和 Top-5 准确率

    返回:
        list[float]: 每个 K 值对应的准确率 (百分比形式, 0.0~100.0)
                     例如 topk=(1,5) 时返回 [top1_acc, top5_acc]

    使用示例:
        logits = model(images)          # [1024, 1000]
        acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
        print(f"Top-1: {acc1:.2f}%  Top-5: {acc5:.2f}%")
    """

    # ── 获取需要的最大 K 值 ──────────────────────────────────────────────────
    # max(topk): 找出 topk 元组中最大的 K 值
    # 例如 topk=(1, 5) → max_k=5
    # 我们只需要对模型输出做一次 topk 运算,取出前 max_k 个预测
    max_k = max(topk)

    # ── 获取 batch 大小 ───────────────────────────────────────────────────────
    # output.size(0): Tensor 第 0 维 (batch 维) 的大小
    # 即这个 batch 中有多少个样本 (通常是 batch_size 或最后一个不完整 batch 的大小)
    batch_size = output.size(0)

    # ── 取出前 max_k 个预测类别 ──────────────────────────────────────────────
    # torch.topk(input, k, dim): 沿指定维度取出最大的 k 个值
    # - output: [N, C] — 每行是一个样本的 1000 维 logits
    # - k=max_k: 取前 max_k 个
    # - dim=1: 沿类别维度 (每一行内) 排序
    # - largest=True (默认): 取最大的 k 个 (最高置信度的类别)
    # - sorted=True (默认): 结果按从大到小排序
    #
    # 返回: (values, indices)
    # - values:  [N, max_k] — 前 max_k 个最高 logit 值 (这里不需要)
    # - pred:    [N, max_k] — 对应的类别索引
    #   pred[i, 0] = 样本 i 的 Top-1 预测类别
    #   pred[i, 1] = 样本 i 的 Top-2 预测类别
    #   ...
    _, pred = output.topk(max_k, dim=1, largest=True, sorted=True)

    # ── 转置预测矩阵,便于后续逐行比较 ──────────────────────────────────────
    # pred: [N, max_k] → pred.t(): [max_k, N]
    # 转置后,pred[k-1, :] 是所有样本的第 k 名预测
    # 这样可以用 expand_as 广播比较
    pred = pred.t()  # [max_k, N]

    # ── 准备真实标签的广播形式 ────────────────────────────────────────────────
    # target: [N] 一维向量
    # target.view(1, -1): [1, N] → 变成二维,第 0 维大小为 1
    # expand_as(pred): [max_k, N] → 将 [1, N] 广播扩展到 [max_k, N]
    # 效果: 将每个样本的真实标签复制 max_k 行,方便与 pred 逐行比较
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    # correct: [max_k, N] 布尔矩阵
    # correct[k, i] = True 表示样本 i 的第 (k+1) 名预测 == 真实标签

    # ── 对每个 K 值计算准确率 ─────────────────────────────────────────────────
    results = []
    for k in topk:
        # correct[:k]: 取前 k 行,即前 k 名预测的命中情况 [k, N]
        #
        # .any(dim=0): 沿第 0 维 (行方向) 做逻辑或
        # 效果: 对每个样本,只要前 k 名中有任意一个预测正确,结果为 True
        # 返回形状: [N]  (每个样本是否 Top-K 命中)
        #
        # .float(): 将布尔 Tensor 转换为浮点数 (True→1.0, False→0.0)
        #
        # .sum(): 求和,得到命中的样本总数 (标量)
        #
        # .item(): 将 PyTorch 标量 Tensor 转换为 Python float
        #
        # / batch_size * 100: 转换为百分比
        correct_k = correct[:k].any(dim=0).float().sum().item()

        # 计算准确率百分比: 命中样本数 / 总样本数 × 100
        acc = correct_k / batch_size * 100.0

        results.append(acc)

    # 返回与 topk 参数顺序一致的准确率列表
    # 例如 topk=(1,5) 时返回 [top1_acc, top5_acc]
    return results


# =============================================================================
# 快速验证入口
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("AverageMeter 测试")
    print("=" * 50)

    # 创建一个用于跟踪 loss 的计数器
    meter = AverageMeter("Loss")

    # 模拟 3 个 batch:
    # batch 1: loss=2.5, 1024 个样本
    # batch 2: loss=2.1, 1024 个样本
    # batch 3: loss=1.8, 576 个样本 (最后一个不完整 batch)
    meter.update(2.5, n=1024)
    meter.update(2.1, n=1024)
    meter.update(1.8, n=576)

    # 手动计算期望均值:
    # (2.5×1024 + 2.1×1024 + 1.8×576) / (1024+1024+576)
    # = (2560 + 2150.4 + 1036.8) / 2624 = 5747.2 / 2624 ≈ 2.1902
    print(f"avg = {meter.avg:.4f}  (期望约 2.1902)")
    print(f"val = {meter.val}  (最近一次值)")
    print(f"count = {meter.count}  (总样本数)")
    print(repr(meter))

    print()
    print("=" * 50)
    print("accuracy 函数测试")
    print("=" * 50)

    # 构造一个 batch: 4 个样本,5 个类别
    # logits[i, j] = 样本 i 对类别 j 的得分
    logits = torch.tensor(
        [
            [0.1, 0.9, 0.3, 0.2, 0.1],  # 样本0: 最高分在类别1
            [0.8, 0.1, 0.3, 0.2, 0.1],  # 样本1: 最高分在类别0
            [0.1, 0.2, 0.9, 0.4, 0.3],  # 样本2: 最高分在类别2
            [0.1, 0.2, 0.3, 0.1, 0.8],  # 样本3: 最高分在类别4
        ]
    )

    # 真实标签: [1, 0, 2, 3]
    # 样本0真实类别=1 (预测也是1, Top-1正确)
    # 样本1真实类别=0 (预测也是0, Top-1正确)
    # 样本2真实类别=2 (预测也是2, Top-1正确)
    # 样本3真实类别=3 (预测是4, Top-1错误; 但类别3在前2名, Top-2正确)
    labels = torch.tensor([1, 0, 2, 3])

    # 计算 Top-1 和 Top-2 准确率
    acc1, acc2 = accuracy(logits, labels, topk=(1, 2))
    # 期望: Top-1 = 3/4 = 75.0%, Top-2 = 4/4 = 100.0%
    print(f"Top-1 准确率: {acc1:.1f}%  (期望 75.0%)")
    print(f"Top-2 准确率: {acc2:.1f}%  (期望 100.0%)")
