# ViT ImageNet 图像分类 — 技术文档

> 本文档面向正在学习深度学习的读者，从工程落地角度详细讲解整个项目的技术选型、架构设计、实现细节和最佳实践。

---

## 目录

1. [项目概述](#1-项目概述)
2. [技术栈总览](#2-技术栈总览)
3. [项目架构与文件职责](#3-项目架构与文件职责)
4. [模型架构详解 (model.py)](#4-模型架构详解-modelpy)
5. [数据处理流水线 (data_processing.py)](#5-数据处理流水线-data_processingpy)
6. [训练流程详解 (train.py)](#6-训练流程详解-trainpy)
7. [测试与推理 (test.py)](#7-测试与推理-testpy)
8. [工具模块 (utils/)](#8-工具模块-utils)
9. [配置系统 (configs/)](#9-配置系统-configs)
10. [关键技术概念深度讲解](#10-关键技术概念深度讲解)
11. [性能优化策略](#11-性能优化策略)
12. [工程最佳实践](#12-工程最佳实践)
13. [常见问题与调试](#13-常见问题与调试)
14. [扩展方向](#14-扩展方向)

---

## 1. 项目概述

### 1.1 项目目标

从零实现 **Vision Transformer (ViT)** 模型，在 **ImageNet** 数据集上完成 1000 类图像分类任务。

### 1.2 论文基础

- **"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"** (Dosovitskiy et al., 2020)
- 核心思想：将图像切成 16×16 的 patch，每个 patch 视为一个 "token"（类比 NLP 中的词），送入标准 Transformer 编码器

### 1.3 为什么选择 ViT 而非传统 CNN？

| 对比维度 | CNN (如 ResNet) | ViT |
|---------|-----------------|-----|
| 感受野 | 逐层扩大（局部→全局） | 第一层就是全局注意力 |
| 归纳偏置 | 强（局部性、平移不变性） | 弱（需更多数据学习） |
| 扩展性 | 加深变难（梯度消失） | 加深更容易 |
| 数据需求 | 较少即可收敛 | 需要大量数据 |
| 计算模式 | 卷积（空间局部）| 自注意力（全局交互）|

---

## 2. 技术栈总览

### 2.1 核心框架与库

| 库 | 版本要求 | 用途 |
|---|---------|------|
| **PyTorch** | ≥ 2.0 | 深度学习框架（张量运算、自动微分、GPU 加速） |
| **torchvision** | ≥ 0.15 | 图像数据集加载（ImageFolder）、数据增强（transforms） |
| **PyYAML** | ≥ 6.0 | 配置文件解析 |
| **Pillow** | ≥ 9.0 | 图像 I/O（被 torchvision 内部调用） |

### 2.2 涉及的工具与技术

本项目涵盖了以下 CNN/ViT 图像分类项目中常见的工具和技术：

#### 模型相关
- **Patch Embedding**: 图像分块嵌入（Conv2d 实现）
- **Multi-Head Self-Attention (MHSA)**: 多头自注意力机制
- **Feed-Forward Network (FFN/MLP)**: 前馈网络
- **Layer Normalization**: 层归一化
- **GELU 激活函数**: Transformer 标准激活
- **Positional Encoding**: 可学习位置编码
- **[CLS] Token**: 分类标记
- **DropPath / Stochastic Depth**: 随机深度正则化
- **Weight Initialization**: 权重初始化策略

#### 训练相关
- **AMP (Automatic Mixed Precision)**: FP16 自动混合精度
- **GradScaler**: 梯度缩放器
- **AdamW 优化器**: 解耦权重衰减
- **参数分组**: 不同参数不同衰减策略
- **Learning Rate Warmup**: 学习率预热
- **Cosine Annealing**: 余弦退火调度
- **Gradient Clipping**: 梯度裁剪
- **Label Smoothing**: 标签平滑
- **Checkpointing**: 断点续训

#### 数据相关
- **ImageFolder**: 标准目录格式数据集
- **Data Augmentation**: 数据增强流水线
- **DataLoader 性能优化**: pin_memory, num_workers, prefetch, persistent_workers

#### 工程相关
- **YAML 配置管理**: 超参数集中管理
- **Python logging**: 日志系统
- **argparse**: 命令行参数解析
- **Top-K Accuracy**: 评估指标

---

## 3. 项目架构与文件职责

```
vit_imagenet_classifier/
│
├── configs/
│   └── default.yaml          # [配置] 所有超参数的单一真相源 (Single Source of Truth)
│
├── dataset/
│   └── imagenet/             # [数据] ImageFolder 格式
│       ├── train/            #   训练集 (按类别子文件夹)
│       ├── val/              #   验证集 (按类别子文件夹)
│       └── synset_words.txt  #   类别ID→名称映射 (可选)
│
├── model.py                  # [模型] ViT 完整实现 (5个类 + 1个工厂函数)
├── data_processing.py        # [数据] 增强流水线 + DataLoader 构建 + 配置加载
├── train.py                  # [训练] 主训练脚本 (优化器 + 调度器 + 训练循环)
├── test.py                   # [测试] 验证评估 + 单图推理
│
├── utils/
│   ├── __init__.py           # [工具] 包初始化 + 统一导出
│   ├── logger.py             # [工具] 日志记录 (console + file)
│   ├── metrics.py            # [工具] 指标计算 (AverageMeter + Top-K Accuracy)
│   └── checkpointing.py      # [工具] 权重保存 + 断点续训
│
├── requirements.txt          # [环境] 依赖列表
├── README.md                 # [文档] 项目说明
└── docs/                     # [文档] 详细文档与流程图
```

### 3.1 模块依赖关系

```
train.py ──────────┬──→ model.py (build_vit)
                   ├──→ data_processing.py (build_dataloaders, load_config)
                   ├──→ utils/logger.py (get_logger)
                   ├──→ utils/metrics.py (AverageMeter, accuracy)
                   └──→ utils/checkpointing.py (save/load_checkpoint)

test.py ───────────┬──→ model.py (build_vit)
                   ├──→ data_processing.py (build_dataloaders, load_config)
                   ├──→ utils/logger.py (get_logger)
                   └──→ utils/metrics.py (AverageMeter, accuracy)

data_processing.py ──→ torchvision (datasets.ImageFolder, transforms)

model.py ────────────→ torch.nn (所有层)
```

---

## 4. 模型架构详解 (model.py)

### 4.1 ViT 整体架构

```
输入图像 [B, 3, 224, 224]
    │
    ▼
┌─────────────────────────┐
│  PatchEmbedding         │  Conv2d(3→768, k=16, s=16)
│  [B, 3, 224, 224]       │  切成 14×14 = 196 个 patch
│  → [B, 196, 768]        │  每个 patch 投影为 768 维向量
└─────────────────────────┘
    │
    ▼ 拼接 [CLS] token
┌─────────────────────────┐
│  [B, 197, 768]          │  cls_token: 可学习参数 [1, 1, 768]
│  = [CLS] + 196 patches  │  expand 到 batch 维度后拼接
└─────────────────────────┘
    │
    ▼ 加上位置编码
┌─────────────────────────┐
│  + pos_embed            │  可学习参数 [1, 197, 768]
│  [B, 197, 768]          │  为每个 token 注入位置信息
└─────────────────────────┘
    │
    ▼ Dropout
    │
    ▼ × 12 层 TransformerBlock
┌─────────────────────────────────────────┐
│  TransformerBlock (Pre-Norm):           │
│  ┌───────────────────────────────────┐  │
│  │ x = x + DropPath(Attn(LN(x)))    │  │  ← 多头自注意力 + 残差
│  │ x = x + DropPath(MLP(LN(x)))     │  │  ← 前馈网络 + 残差
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
    │
    ▼ LayerNorm
    │
    ▼ 取 [CLS] token (x[:, 0])
┌─────────────────────────┐
│  [B, 768]               │  CLS token 聚合了全局信息
└─────────────────────────┘
    │
    ▼ Linear(768 → 1000)
┌─────────────────────────┐
│  [B, 1000] logits       │  分类输出
└─────────────────────────┘
```

### 4.2 PatchEmbedding 详解

**核心思想**: 将图像视为一个 "patch 序列"，类比 NLP 中的 "词序列"。

```python
# 等价操作对比：
# 方法 A（概念清晰但低效）：
patches = image.unfold(2, 16, 16).unfold(3, 16, 16)  # 手动切块
patches = patches.reshape(B, -1, 3*16*16)             # 展平
embeddings = linear(patches)                           # 线性投影

# 方法 B（本项目使用，高效）：
embeddings = Conv2d(in_channels=3, out_channels=768, kernel_size=16, stride=16)(image)
# kernel_size = stride = patch_size → 无重叠切割
# out_channels = embed_dim → 直接投影到嵌入空间
```

**为什么用 Conv2d？**
- GPU 对卷积操作有深度优化（cuDNN）
- 一步完成 "切块 + 线性投影"
- 结果数学上完全等价

### 4.3 Multi-Head Self-Attention 详解

**直觉**: 让每个 patch 能 "看到" 并 "关注" 所有其他 patch。

**数学过程**:

1. **线性投影**: 生成 Query (Q), Key (K), Value (V)
   ```
   Q = X·W_Q,  K = X·W_K,  V = X·W_V
   ```
   本项目用一个合并的 Linear 层: `qkv = Linear(D, 3D)`

2. **分头**: 将 768 维拆成 12 个 64 维的 "头"
   ```
   每个头独立计算注意力，关注不同的特征子空间
   ```

3. **计算注意力分数**:
   ```
   Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
   ```
   - `Q·K^T`: 计算 query 与所有 key 的相似度
   - `/ √d_k`: 缩放，防止点积值过大导致 softmax 梯度消失
   - `softmax`: 转为概率分布（每行和为 1）
   - `· V`: 用注意力权重加权聚合 value

4. **合并多头 + 输出投影**:
   ```
   MultiHead = Concat(head_1, ..., head_12) · W_O
   ```

**时间复杂度**: O(N²·D)，其中 N=197（序列长度），D=768。这是 ViT 的计算瓶颈。

### 4.4 MLP (Feed-Forward Network)

```
x → Linear(768→3072) → GELU → Dropout → Linear(3072→768) → Dropout
```

- **4 倍扩展**: 隐藏层维度 = embed_dim × mlp_ratio = 768 × 4 = 3072
- **GELU 激活**: 比 ReLU 更平滑，是 Transformer 的标准选择
  - GELU(x) = x · Φ(x)，其中 Φ 是标准正态分布的 CDF
  - 在 x≈0 附近是平滑的（ReLU 在 0 处不可微）

### 4.5 Pre-Norm vs Post-Norm

本项目使用 **Pre-Norm** 结构（LayerNorm 在子层之前）:

```
Pre-Norm:   x = x + Sublayer(LN(x))    ← 本项目
Post-Norm:  x = LN(x + Sublayer(x))    ← 原始 Transformer
```

**Pre-Norm 的优势**:
- 训练更稳定，对学习率不那么敏感
- 梯度流更通畅（残差路径上没有 LN）
- 是 ViT 原论文的选择

### 4.6 DropPath (Stochastic Depth)

```python
# 训练时：以概率 drop_prob 将整条残差路径置零
if training:
    mask = bernoulli(keep_prob)  # 0 或 1，per-sample
    output = x / keep_prob * mask  # 缩放以保持期望值
```

- **线性递增策略**: 浅层 drop=0（特征提取关键），深层逐渐增大到 max_drop_path=0.1
- **效果**: 相当于隐式训练了多个不同深度的子网络（类似 Dropout 的集成效果）

### 4.7 权重初始化

```python
# ViT 论文推荐的初始化方案:
- 位置编码、CLS token: truncated_normal(std=0.02)
- Linear 层 weight:     truncated_normal(std=0.02)
- Linear 层 bias:       zeros
- LayerNorm weight:      ones
- LayerNorm bias:        zeros
- Conv2d weight:         kaiming_normal (fan_out)
```

**为什么初始化很重要？**
- 不良初始化 → 训练不收敛或收敛极慢
- 截断正态分布避免极端值，保持激活值在合理范围
- Kaiming 初始化考虑了激活函数的方差缩放

---

## 5. 数据处理流水线 (data_processing.py)

### 5.1 训练集数据增强

数据增强的目的是 **人工增加训练数据的多样性**，防止过拟合。

```
原始图像 (任意尺寸)
    │
    ▼ RandomResizedCrop(224, scale=(0.08, 1.0))
    │  随机裁剪一块区域（面积占 8%~100%），resize 到 224×224
    │  效果：强制模型从局部特征推断类别
    │
    ▼ RandomHorizontalFlip(p=0.5)
    │  50% 概率水平翻转
    │  效果：猫向左看和向右看都是猫
    │
    ▼ ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
    │  随机调整亮度、对比度、饱和度、色调
    │  效果：模拟不同光照和相机条件
    │
    ▼ ToTensor()
    │  PIL Image (HWC, uint8, [0,255]) → Tensor (CHW, float32, [0,1])
    │
    ▼ Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
       output[c] = (input[c] - mean[c]) / std[c]
       效果：输入分布 → 近似 N(0,1)，加速收敛
```

### 5.2 验证集变换

```
原始图像 → Resize(256) → CenterCrop(224) → ToTensor() → Normalize(同上)
```

- **无随机操作**: 保证评估结果可复现
- **Resize(256) + CenterCrop(224)**: 业界标准验证方式

### 5.3 ImageFolder 数据集格式

```
dataset/imagenet/
├── train/
│   ├── n01440764/    → 类别 0 (tench 丁鲷)
│   │   ├── ILSVRC2012_val_00000293.JPEG
│   │   └── ...
│   ├── n01443537/    → 类别 1 (goldfish 金鱼)
│   └── ...           → 共 1000 个类别文件夹
└── val/
    ├── n01440764/
    └── ...
```

`torchvision.datasets.ImageFolder` 自动：
1. 扫描所有子文件夹
2. 按字母序将文件夹名映射为整数标签 (0, 1, 2, ...)
3. 返回 (image_tensor, label_int) 格式

### 5.4 DataLoader 性能优化

| 参数 | 值 | 作用 | 原理 |
|-----|-----|------|------|
| `pin_memory` | True | 加速 CPU→GPU 传输 | 锁页内存，GPU DMA 直接读取，绕过内核拷贝 |
| `num_workers` | 8 | 并行数据加载 | 多进程同时解码图片 + 执行 transform |
| `persistent_workers` | True | 保持 worker 存活 | 避免每 epoch 重建进程池 (数秒开销) |
| `prefetch_factor` | 2 | 预取 2 个 batch | GPU 处理当前 batch 时，下一批已在内存就绪 |
| `drop_last` | True(train) | 丢弃不完整 batch | 保证 BatchNorm 统计稳定 |
| `shuffle` | True(train) | 每 epoch 打乱 | 防止模型记住样本顺序 |
| `non_blocking` | True | 异步传输 | `.to(device, non_blocking=True)` 不阻塞 CPU |

---

## 6. 训练流程详解 (train.py)

### 6.1 训练主函数 main() 流程

```
1. 加载配置 (YAML → dict)
2. 创建 logger (console + file)
3. 检测设备 (CUDA GPU / CPU)
4. 构建数据 (DataLoader)
5. 构建模型 (ViT → GPU)
6. 定义损失函数 (CrossEntropyLoss + Label Smoothing)
7. 构建优化器 (AdamW + 参数分组)
8. 构建学习率调度器 (Warmup + Cosine)
9. 初始化 AMP GradScaler
10. [可选] 从 checkpoint 恢复
11. 训练循环 (train → validate → save)
12. 输出最终结果
```

### 6.2 AdamW 优化器与参数分组

**为什么 ViT 用 AdamW 而非 SGD？**
- Transformer 的 loss landscape 比 CNN 更不平坦
- Adam 的自适应学习率能更好地导航复杂的损失面
- AdamW 将权重衰减从梯度更新中解耦，正则化效果更好

**参数分组策略**:
```python
# 施加权重衰减 (weight_decay=0.05)
decay_params = [普通权重矩阵]

# 不施加权重衰减 (weight_decay=0.0)
no_decay_params = [
    包含 "norm" 的参数,     # LayerNorm 参数不应被正则化
    包含 "bias" 的参数,     # 偏置项
    包含 "cls_token" 的参数, # CLS 标记
    包含 "pos_embed" 的参数, # 位置编码
]
```

**为什么要分组？**
- LayerNorm 的 γ 和 β 用于控制特征分布，weight_decay 会干扰其功能
- Bias 项本身维度低，不需要正则化
- CLS token 和位置编码是特殊的可学习参数，不应被衰减

### 6.3 学习率调度: Warmup + Cosine Annealing

```
学习率变化曲线:

   lr │        ╱─────────╲
      │      ╱             ╲
      │    ╱                 ╲
      │  ╱                     ╲
      │╱                         ╲_____ min_lr
      └────┬──────────────────────┬──→ epoch
           20                    300
        Warmup        CosineAnnealing
```

**Warmup 阶段 (前 20 个 epoch)**:
- 学习率从 `lr × 1e-4` 线性增长到 `lr`
- 原因：训练初期模型权重随机，大学习率容易导致梯度爆炸

**CosineAnnealing 阶段 (后 280 个 epoch)**:
- 学习率从 `lr` 平滑余弦衰减到 `min_lr (1e-5)`
- 原因：训练后期需要更小的学习率精细调整

### 6.4 AMP 混合精度训练

```
前向传播:
    with autocast():              # 自动将 FP32→FP16
        logits = model(images)    # 模型计算用 FP16 (快2倍)
        loss = criterion(...)     # 损失计算用 FP32 (精度需要)

反向传播:
    scaler.scale(loss).backward() # 缩放 loss 防止 FP16 梯度下溢
    scaler.unscale_(optimizer)    # 反缩放，恢复真实梯度值
    clip_grad_norm_(...)          # 梯度裁剪 (必须在 unscale 之后)
    scaler.step(optimizer)        # 用缩放后的梯度更新参数
    scaler.update()               # 动态调整缩放因子
```

**为什么用混合精度？**
- **速度**: FP16 计算在 Tensor Cores 上快 2-3 倍
- **显存**: FP16 激活值显存减半，可用更大 batch_size
- **精度**: 关键操作（损失计算、梯度累积）仍用 FP32，几乎不损失精度

### 6.5 梯度裁剪

```python
if max_grad_norm is not None:
    scaler.unscale_(optimizer)  # 必须先反缩放
    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # max_norm=1.0
```

**为什么 ViT 需要梯度裁剪？**
- Transformer 训练早期，注意力分数可能出现极端值
- 导致梯度爆炸（loss 突然变 NaN）
- clip_grad_norm_ 将所有参数的梯度范数缩放到 max_norm 以内

### 6.6 Label Smoothing

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

**原理**: 将 one-hot 标签 `[0, 0, 1, 0, 0]` 平滑为 `[0.025, 0.025, 0.9, 0.025, 0.025]`

```
平滑后: y_smooth = (1 - ε) * y_onehot + ε / K
其中 ε=0.1, K=1000(类别数)
```

**效果**:
- 防止模型过度自信（输出概率不会趋向 0/1 极端值）
- 隐式正则化，提升泛化能力
- 是 ViT 训练的标准配置

---

## 7. 测试与推理 (test.py)

### 7.1 验证集评估模式

```
python test.py --config configs/default.yaml --checkpoint checkpoints/best_model.pth
```

流程:
1. 加载配置和模型权重
2. `model.eval()` — 关闭 Dropout 和 BatchNorm 更新
3. 构建验证集 DataLoader
4. `@torch.no_grad()` — 禁用梯度计算（省显存、加速）
5. 遍历验证集，计算 Loss/Top-1/Top-5
6. 输出最终指标

### 7.2 单图推理模式

```
python test.py --checkpoint checkpoints/best_model.pth --image path/to/cat.jpg
```

流程:
1. 加载模型权重
2. 使用与验证集相同的 transform 处理图片
3. `unsqueeze(0)` 添加 batch 维度: [3,224,224] → [1,3,224,224]
4. 前向推理得到 logits
5. softmax → 概率分布
6. topk(5) → Top-5 预测
7. 如果有 synset_words.txt → 输出可读类名

### 7.3 Top-K Accuracy 的含义

- **Top-1**: 模型预测的最可能类别就是正确答案的概率
- **Top-5**: 正确答案在模型预测的前 5 个类别中的概率
- ImageNet 基准通常报告这两个指标
- ViT-Base/16 在 ImageNet-1K 上通常能达到 ~77% Top-1

---

## 8. 工具模块 (utils/)

### 8.1 logger.py — 日志系统

```python
logger = get_logger(name="train", log_file="logs/train.log")
logger.info("训练开始")   # 同时输出到控制台和文件
logger.debug("调试信息")  # 只写入文件，控制台不显示
```

**设计要点**:
- **Console Handler**: level=INFO，实时查看关键信息
- **File Handler**: level=DEBUG，记录所有细节供事后分析
- **防重复**: 检查 `logger.handlers` 避免多次调用时重复添加
- **编码**: `encoding='utf-8'` 支持中文日志

### 8.2 metrics.py — 指标计算

#### AverageMeter
```python
meter = AverageMeter("Loss")
meter.update(val=0.5, n=32)   # 32 个样本的平均 loss 是 0.5
meter.update(val=0.3, n=32)   # 又来 32 个样本
print(meter.avg)               # → 0.4 (加权平均)
```

- 使用加权平均（而非简单平均），因为最后一个 batch 可能不满

#### accuracy 函数
```python
acc1, acc5 = accuracy(output, target, topk=(1, 5))
# output: [N, 1000] 模型输出
# target: [N] 真实标签
# 返回: [top1_百分比, top5_百分比]
```

### 8.3 checkpointing.py — 断点续训

**保存的完整状态**:
```python
state = {
    "epoch": next_epoch,             # 下一个要训练的 epoch
    "model_state_dict": ...,         # 模型所有参数
    "optimizer_state_dict": ...,     # 优化器状态 (动量、二阶矩)
    "scheduler_state_dict": ...,     # 调度器状态 (当前在哪个阶段)
    "scaler_state_dict": ...,        # AMP 缩放器状态
    "val_top1": ...,                 # 当前验证精度
    "best_top1": ...,                # 历史最优精度
}
```

**为什么要保存优化器状态？**
- AdamW 为每个参数维护一阶矩 (m) 和二阶矩 (v)
- 如果只恢复模型权重，优化器从零开始 → 训练出现明显的 "跳变"
- 恢复完整状态 → 训练无缝继续

**两个文件**:
- `latest.pth`: 每个 epoch 覆盖保存（断点续训用）
- `best_model.pth`: 仅在验证精度创新高时保存（最终推理用）

---

## 9. 配置系统 (configs/)

### 9.1 default.yaml 结构

```yaml
data:
  root: "dataset/imagenet"    # 数据集路径
  batch_size: 128             # 训练 batch size
  val_batch_size: 256         # 验证 batch size (无梯度，可更大)
  num_workers: 8              # 数据加载进程数
  prefetch_factor: 2          # 预取倍数

model:
  name: "vit_base_patch16"    # 模型标识
  img_size: 224               # 输入分辨率
  patch_size: 16              # Patch 大小
  num_classes: 1000           # 类别数
  embed_dim: 768              # 嵌入维度
  depth: 12                   # Transformer 层数
  num_heads: 12               # 注意力头数
  mlp_ratio: 4.0              # MLP 扩展倍数

train:
  epochs: 300                 # 总训练轮数
  lr: 1.0e-3                  # 基础学习率
  weight_decay: 0.05          # 权重衰减
  warmup_epochs: 20           # Warmup 轮数
  min_lr: 1.0e-5              # 最低学习率
  label_smoothing: 0.1        # 标签平滑
  max_grad_norm: 1.0          # 梯度裁剪阈值
```

### 9.2 配置管理最佳实践

1. **单一真相源**: 所有超参数集中在 YAML 文件中，代码中不硬编码
2. **安全解析**: 使用 `yaml.safe_load` 而非 `yaml.load`，防止代码注入
3. **默认值**: 代码中用 `.get(key, default)` 提供默认值，增强鲁棒性
4. **实验管理**: 修改配置即可开始新实验，无需改代码

---

## 10. 关键技术概念深度讲解

### 10.1 Tensor Shapes 变化追踪

这是理解整个模型最重要的线索之一：

```
输入:              [B, 3, 224, 224]     # Batch × RGB × Height × Width

PatchEmbedding:
  Conv2d:          [B, 768, 14, 14]     # 768 个 feature map
  flatten(2):      [B, 768, 196]        # 展平空间维度
  transpose(1,2):  [B, 196, 768]        # 序列格式: 196 个 token

拼接 CLS:         [B, 197, 768]         # +1 CLS token

位置编码:         [B, 197, 768]         # 元素相加 (广播)

Attention:
  QKV 投影:        [B, 197, 2304]       # 768 × 3
  reshape:         [B, 197, 3, 12, 64]  # 3(QKV) × 12头 × 64维/头
  permute:         [3, B, 12, 197, 64]  # 分离 Q, K, V
  attn scores:     [B, 12, 197, 197]    # 197×197 注意力矩阵
  attn × V:        [B, 12, 197, 64]     # 加权聚合
  concat heads:    [B, 197, 768]        # 12×64=768

MLP:
  fc1:             [B, 197, 3072]       # 768 × 4 = 3072
  fc2:             [B, 197, 768]        # 压缩回 768

× 12 层后:        [B, 197, 768]

取 CLS:           [B, 768]             # x[:, 0]
分类头:           [B, 1000]            # Linear(768→1000)
```

### 10.2 自注意力的直觉理解

想象你在看一张猫的图片：

1. **Query** = "这个 patch 在找什么信息？"
2. **Key** = "这个 patch 包含什么信息？"
3. **Value** = "这个 patch 的实际内容"
4. **注意力分数** = Query 和 Key 的匹配程度
5. **输出** = 所有 Value 的加权混合（权重 = 注意力分数）

例如：一个包含猫耳朵的 patch (Query) 可能会高度关注 (高 attention score) 另一个包含猫眼睛的 patch (Key)，因为它们共同构成了 "猫脸" 的特征。

### 10.3 残差连接的重要性

```python
x = x + self.attn(self.norm(x))  # 残差连接
```

**为什么不直接 x = self.attn(self.norm(x))？**
- 12 层串联，梯度需要回传到第 1 层
- 没有残差 → 梯度经过 12 次变换 → 可能消失或爆炸
- 有残差 → 梯度可以直接 "穿越" 残差路径到达任意层
- 残差连接是深层网络训练的基础保障

### 10.4 为什么需要位置编码？

```python
self.pos_embed = nn.Parameter(torch.zeros(1, 197, 768))
```

- Self-Attention 是 **排列不变** 的: 打乱 token 顺序，输出不变
- 但图像有空间结构: 左上角的 patch 和右下角的 patch 位置不同
- 位置编码为每个 token 注入 "我在第几个位置" 的信息
- 本项目使用 **可学习位置编码** (而非 Sinusoidal)，让模型自己学习最优的位置表示

---

## 11. 性能优化策略

### 11.1 GPU 利用率优化

| 技术 | 效果 | 原理 |
|-----|------|------|
| AMP (FP16) | 训练速度 ×2, 显存 ×0.5 | Tensor Cores 加速 + 半精度存储 |
| cudnn.benchmark | 卷积速度 +10~20% | 自动搜索最快的卷积算法 |
| non_blocking=True | 隐藏 CPU→GPU 延迟 | 异步数据传输 |
| zero_grad(set_to_none=True) | 节省一次 memset | 不清零，直接释放梯度内存 |

### 11.2 数据加载优化

| 技术 | 效果 | 原理 |
|-----|------|------|
| num_workers=8 | 加载速度 ×4~8 | 多进程并行解码 |
| pin_memory=True | CPU→GPU 传输速度 ×2 | DMA 直传 |
| persistent_workers | 省去进程重建开销 | 每 epoch 省 3-5 秒 |
| prefetch_factor=2 | 隐藏 I/O 延迟 | 提前准备好下一个 batch |

### 11.3 数值稳定性

| 技术 | 防止什么问题 | 机制 |
|-----|-------------|------|
| GradScaler | FP16 梯度下溢 | 放大 loss → 放大梯度 → 更新时缩放回来 |
| Gradient Clipping | 梯度爆炸 | 将梯度范数限制在阈值内 |
| Attention Scaling | softmax 梯度消失 | 除以 √d_k 控制点积值范围 |
| Label Smoothing | 过拟合 | 防止输出概率趋向 0/1 极端 |

---

## 12. 工程最佳实践

### 12.1 代码组织原则

1. **关注点分离**: model.py (模型) / data_processing.py (数据) / train.py (训练) / utils/ (工具)
2. **配置与代码分离**: 超参数在 YAML 中，不在代码里硬编码
3. **工厂模式**: `build_vit(cfg)` 从配置字典构建模型，方便切换不同规模
4. **渐进式验证**: 每个文件都有 `if __name__ == "__main__"` 快速测试

### 12.2 可复现性保障

- 验证集使用确定性 transform (无随机操作)
- 完整 checkpoint 保存 (模型+优化器+调度器+scaler)
- 配置文件记录所有超参数
- 日志文件记录完整训练过程

### 12.3 错误处理

- 数据目录不存在 → 清晰的 FileNotFoundError 提示
- 配置文件不存在 → 明确的错误消息
- CUDA 不可用 → 警告并回退到 CPU
- yaml.safe_load → 防止配置文件中的代码注入

---

## 13. 常见问题与调试

### 13.1 显存不足 (CUDA OOM)

```
RuntimeError: CUDA out of memory
```

**解决方案**:
1. 减小 `data.batch_size` (128 → 64 → 32)
2. 确认 AMP 已启用 (GradScaler)
3. 使用更小的模型 (减小 embed_dim 或 depth)

### 13.2 Loss 不下降

**检查清单**:
1. 学习率是否合适？(尝试 1e-4 ~ 1e-3)
2. 数据增强是否正确？(打印一个 batch 检查)
3. 标签是否正确？(ImageFolder 自动映射)
4. Warmup 是否足够？(尝试增加 warmup_epochs)

### 13.3 训练中 Loss 变成 NaN

**可能原因**:
1. 学习率过大 → 减小 lr
2. 梯度爆炸 → 确认 max_grad_norm 已设置
3. AMP 数值问题 → GradScaler 会自动处理（skip step）

---

## 14. 扩展方向

### 14.1 进阶技术

| 技术 | 说明 | 难度 |
|-----|------|------|
| **DeiT** (Data-efficient Image Transformers) | 知识蒸馏 + 更强数据增强 | ★★★ |
| **Swin Transformer** | 窗口注意力 + 层级结构 | ★★★★ |
| **MAE** (Masked Autoencoder) | 自监督预训练 | ★★★★ |
| **MixUp / CutMix** | 更强的数据增强 | ★★ |
| **RandAugment / AutoAugment** | 自动数据增强策略 | ★★ |
| **EMA** (Exponential Moving Average) | 权重指数移动平均 | ★★ |
| **分布式训练** (DDP) | 多卡并行训练 | ★★★ |
| **TensorBoard / W&B** | 训练可视化 | ★ |
| **ONNX 导出** | 模型部署 | ★★ |

### 14.2 论文阅读推荐

1. **ViT**: "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)
2. **Attention Is All You Need** (Vaswani et al., 2017) — Transformer 原论文
3. **DeiT**: "Training data-efficient image transformers" (Touvron et al., 2021)
4. **Swin Transformer**: "Hierarchical Vision Transformer" (Liu et al., 2021)
5. **ResNet**: "Deep Residual Learning" (He et al., 2016) — 残差连接的源头

### 14.3 项目迭代建议

1. **先用小数据集验证**: ImageNet-100 (100 类) 或 CIFAR-10
2. **添加训练可视化**: TensorBoard 记录 loss/acc 曲线
3. **实现更多增强**: MixUp, CutMix, RandAugment
4. **尝试预训练微调**: 加载 timm 预训练权重
5. **实现分布式训练**: PyTorch DDP
6. **模型部署**: ONNX 导出 + TorchScript

---

> 文档版本: v1.0
> 生成日期: 2026-04-01
> 对应代码: ViT-Base/16 ImageNet Classifier
