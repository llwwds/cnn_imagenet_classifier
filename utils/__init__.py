# =============================================================================
# utils/__init__.py — utils 工具包的公共接口声明
# =============================================================================
#
# 什么是 __init__.py?
#   __init__.py 是 Python "包" (Package) 的标志文件。
#   一个目录只要包含 __init__.py,Python 就将其视为一个"包",
#   可以被其他模块通过 import 语句导入。
#
#   没有 __init__.py 的目录只是普通文件夹,无法被 import。
#
# 这个文件的作用:
#   将 utils 子模块中最常用的函数/类"提升"到 utils 包的顶层命名空间。
#
#   有了这里的导入声明:
#     # 简洁写法 (推荐):
#     from utils import get_logger, AverageMeter, accuracy
#     from utils import save_checkpoint, load_checkpoint
#
#   没有这里的导入声明时,只能用完整路径:
#     from utils.logger       import get_logger
#     from utils.metrics      import AverageMeter, accuracy
#     from utils.checkpointing import save_checkpoint, load_checkpoint
#
# =============================================================================

# ── 从 logger 模块导入日志记录器工厂函数 ──────────────────────────────────────
# get_logger: 创建一个同时输出到控制台和文件的日志记录器
from utils.logger import get_logger

# ── 从 metrics 模块导入训练指标工具 ───────────────────────────────────────────
# AverageMeter: 跟踪并累积计算均值的计数器 (用于 loss、accuracy 等)
# accuracy:     计算 Top-K 准确率的函数
from utils.metrics import AverageMeter, accuracy

# ── 从 checkpointing 模块导入 Checkpoint 工具 ─────────────────────────────────
# save_checkpoint: 将完整训练状态保存到磁盘
# load_checkpoint: 从磁盘恢复训练状态,支持断点续训
from utils.checkpointing import save_checkpoint, load_checkpoint

# ── 定义包的公开 API ──────────────────────────────────────────────────────────
# __all__ 是一个特殊变量,定义 "from utils import *" 时会导出哪些名称
# 明确列出所有公开 API,防止意外导出内部实现细节
__all__ = [
    "get_logger",        # 日志记录器
    "AverageMeter",      # 均值计数器
    "accuracy",          # Top-K 准确率
    "save_checkpoint",   # 保存检查点
    "load_checkpoint",   # 加载检查点
]
