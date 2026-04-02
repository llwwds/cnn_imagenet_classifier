# =============================================================================
# utils/logger.py — 日志记录器模块
# 作用: 统一管理训练和测试过程中的日志输出
# 功能:
#   1. 将日志同时输出到控制台 (Console) 和磁盘文件 (File)
#   2. 格式化日志 (时间戳 + 级别 + 消息)
#   3. 防止重复添加 Handler (多次调用 get_logger 时不产生重复输出)
# =============================================================================
#
# 什么是日志 (Logging)?
#   print() 函数虽然简单,但有很多缺点:
#     - 无法区分信息级别 (普通信息 vs 警告 vs 错误)
#     - 无法自动保存到文件
#     - 多线程/多进程场景下可能乱序
#   Python 的 logging 模块解决了这些问题,是生产环境的标准做法。
#
# 日志级别 (从低到高):
#   DEBUG    → 详细的调试信息 (开发时用)
#   INFO     → 普通运行信息 (训练进度、参数设置等)
#   WARNING  → 警告 (GPU 未找到等,程序仍能运行)
#   ERROR    → 错误 (文件不存在等,导致局部功能失效)
#   CRITICAL → 严重错误 (程序无法继续运行)
# =============================================================================

# logging: Python 标准库的日志模块,无需额外安装
import logging

# os: 操作系统接口,用于文件路径操作和目录创建
import os

# sys: 系统接口,用于获取标准输出流 (sys.stdout)
import sys


# =============================================================================
# 工厂函数: get_logger — 创建并返回一个配置好的日志记录器
# =============================================================================


def get_logger(name: str, log_file: str = None) -> logging.Logger:
    """
    创建或获取一个命名的日志记录器,同时输出到控制台和文件。

    Python logging 系统使用"命名空间"来管理 Logger。
    同一个 name 多次调用 get_logger 会返回同一个 Logger 对象
    (Logger 是单例模式按 name 缓存的)。

    参数:
        name     (str):           日志记录器的名称,如 "train"、"test"
                                  用于在多个模块中区分不同来源的日志
        log_file (str, optional): 日志文件路径,如 "logs/train.log"
                                  如果为 None,则只输出到控制台不写文件

    返回:
        logging.Logger: 配置好的日志记录器对象

    使用示例:
        logger = get_logger("train", "logs/train.log")
        logger.info("训练开始")       # 同时输出到控制台和 logs/train.log
        logger.warning("GPU 未找到")  # 带 WARNING 前缀
        logger.error("文件不存在")    # 带 ERROR 前缀
    """

    # ── 获取 Logger 对象 ──────────────────────────────────────────────────────
    # logging.getLogger(name) 是工厂方法:
    #   - 如果 name 对应的 Logger 已存在 → 直接返回缓存的对象 (不重复创建)
    #   - 如果不存在 → 创建一个新的 Logger 对象
    # 这样保证同一个 name 在整个程序运行期间只有一个 Logger 实例
    logger = logging.getLogger(name)

    # ── 防止重复添加 Handler ──────────────────────────────────────────────────
    # logger.handlers 是一个列表,存储了已添加到该 Logger 的所有 Handler
    # 如果列表非空,说明这个 Logger 已经配置过了,直接返回,避免重复输出
    # 重复添加 Handler 的后果: 同一条日志会被打印多次,非常混乱
    if logger.handlers:
        # 已经配置过,直接返回现有 Logger
        return logger

    # ── 设置日志级别 ──────────────────────────────────────────────────────────
    # setLevel(logging.DEBUG): 设置 Logger 接受的最低日志级别
    # 这里设为 DEBUG,表示所有级别的日志都会被接受
    # Handler 层面可以进一步过滤级别 (这里 Handler 也设为 DEBUG,全部输出)
    logger.setLevel(logging.DEBUG)

    # ── 创建日志格式化器 ──────────────────────────────────────────────────────
    # Formatter 定义每条日志的输出格式
    # %(asctime)s    → 时间戳,格式由 datefmt 参数控制
    # %(name)s       → Logger 的名称 (即传入的 name 参数)
    # %(levelname)s  → 日志级别名称 (INFO, WARNING, ERROR 等)
    # %(message)s    → 实际的日志内容 (调用 logger.info(xxx) 时传入的字符串)
    #
    # 示例输出:
    #   [2024-01-15 14:23:45] [train] [INFO] Epoch [1] Loss: 2.3456
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        # datefmt: 时间戳的具体格式
        # %Y=四位年份, %m=两位月份, %d=两位日期, %H=24小时制小时, %M=分钟, %S=秒
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── 创建并配置控制台 Handler ──────────────────────────────────────────────
    # StreamHandler: 将日志输出到一个流 (stream)
    # sys.stdout: 标准输出流 (即终端/控制台)
    # 效果: 训练时可以在终端实时看到日志输出
    console_handler = logging.StreamHandler(sys.stdout)

    # 为控制台 Handler 设置格式化器 (使用上面创建的 formatter)
    console_handler.setFormatter(formatter)

    # 设置控制台 Handler 的最低级别: DEBUG 表示所有级别的日志都显示
    console_handler.setLevel(logging.DEBUG)

    # 将控制台 Handler 添加到 Logger
    # addHandler 之后,每次调用 logger.info/warning/error 时,
    # 日志会被发送到这个 Handler 进行处理 (输出到控制台)
    logger.addHandler(console_handler)

    # ── 创建并配置文件 Handler (如果指定了文件路径) ───────────────────────────
    if log_file is not None:
        # 自动创建日志文件所在的目录 (如果不存在)
        # os.path.dirname("logs/train.log") → "logs"
        # exist_ok=True: 如果目录已存在,不报错
        log_dir = os.path.dirname(log_file)
        if log_dir:
            # 只有当日志路径包含目录部分时才创建目录
            os.makedirs(log_dir, exist_ok=True)

        # FileHandler: 将日志写入磁盘文件
        # mode="a": 追加模式 (append),不覆盖已有内容,适合断点续训时日志连续记录
        # encoding="utf-8": 支持中文和特殊字符
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")

        # 为文件 Handler 设置相同的格式化器
        file_handler.setFormatter(formatter)

        # 文件日志级别也设为 DEBUG (记录所有级别)
        file_handler.setLevel(logging.DEBUG)

        # 将文件 Handler 添加到 Logger
        logger.addHandler(file_handler)

    # ── 禁止日志向父 Logger 传播 ──────────────────────────────────────────────
    # Python logging 系统是树状结构: 每个 Logger 都有一个 parent Logger
    # 默认情况下,日志会向上传播到根 Logger (root logger),可能导致重复输出
    # propagate=False: 阻止传播,日志只由当前 Logger 的 Handler 处理
    logger.propagate = False

    return logger


# =============================================================================
# 快速验证入口
# =============================================================================

if __name__ == "__main__":
    # 创建一个测试用的 Logger,同时输出到控制台和文件
    logger = get_logger("test_logger", log_file="logs/test.log")

    # 测试不同级别的日志输出
    logger.debug("这是 DEBUG 级别的消息 (调试信息)")
    logger.info("这是 INFO 级别的消息 (普通运行信息)")
    logger.warning("这是 WARNING 级别的消息 (警告)")
    logger.error("这是 ERROR 级别的消息 (错误)")

    # 验证同一 name 多次调用不会重复
    # 以下调用会返回同一个 Logger 对象 (不会添加额外 Handler)
    logger2 = get_logger("test_logger")
    logger2.info("第二次获取同一个 Logger (不会重复输出)")

    print("\n验证完成! 查看 logs/test.log 确认文件写入。")
