# CLAUDE.md — ViT ImageNet Classifier 项目参考

> 此文档供 Claude 在后续对话中快速理解项目全貌。

## 项目概述

从零实现 Vision Transformer (ViT-Base/16) 对 ImageNet 进行图像分类。
用户是正在学习 ML/DL 的大一 CS 学生，目标成为 AI 算法工程师/架构师。

## 技术栈

| 组件 | 技术 |
|------|------|
| 框架 | PyTorch >= 2.0 |
| 模型 | ViT-Base/16 (86M params, 12层, 768维, 12头) |
| 混合精度 | torch.cuda.amp (autocast + GradScaler) |
| 优化器 | AdamW (带参数分组: decay / no-decay) |
| 学习率调度 | LinearLR Warmup → CosineAnnealingLR (SequentialLR) |
| 数据增强 | RandomResizedCrop, HFlip, ColorJitter, ImageNet Normalize |
| 正则化 | Label Smoothing (0.1), DropPath/Stochastic Depth, Weight Decay |
| 梯度裁剪 | clip_grad_norm_ (max_norm=1.0) |
| 配置管理 | YAML (configs/default.yaml) |
| 日志 | Python logging (console + file) |

## 文件结构与职责

```
├── configs/default.yaml      ← 所有超参数集中配置
├── model.py                  ← ViT 完整实现 (PatchEmbedding → Attention → MLP → TransformerBlock → VisionTransformer)
├── data_processing.py        ← 数据增强流水线 + ImageFolder 加载 + 高性能 DataLoader
├── train.py                  ← 训练主脚本 (优化器构建 → 单epoch训练/验证 → 主循环)
├── test.py                   ← 测试/推理 (验证集评估 + 单图推理)
├── utils/
│   ├── __init__.py           ← 统一导出
│   ├── logger.py             ← get_logger(): console + file handler
│   ├── metrics.py            ← AverageMeter + accuracy(topk)
│   └── checkpointing.py      ← save_checkpoint / load_checkpoint
└── requirements.txt          ← torch, torchvision, pyyaml, pillow
```

## 模块调用关系

### 训练流 (train.py → main)
```
main(config_path)
 ├── load_config()                          ← data_processing.py
 ├── get_logger()                           ← utils/logger.py
 ├── build_dataloaders(cfg)                 ← data_processing.py
 │   ├── build_datasets(data_root)
 │   │   ├── build_train_transform()
 │   │   └── build_val_transform()
 │   └── DataLoader × 2
 ├── build_vit(cfg)                         ← model.py
 │   └── VisionTransformer(...)
 │       ├── PatchEmbedding
 │       ├── TransformerBlock × depth
 │       │   ├── Attention (Multi-Head Self-Attention)
 │       │   ├── MLP (FFN)
 │       │   └── DropPath
 │       └── Classification Head
 ├── build_optimizer(model, cfg)            ← train.py
 ├── build_scheduler(optimizer, cfg)        ← train.py
 ├── GradScaler()                           ← torch.cuda.amp
 ├── load_checkpoint() [可选]               ← utils/checkpointing.py
 └── for epoch in range:
     ├── train_one_epoch(...)               ← train.py
     │   └── AMP forward → backward → clip_grad → scaler.step
     ├── validate(...)                      ← train.py
     ├── scheduler.step()
     └── save_checkpoint(...)               ← utils/checkpointing.py
```

### 测试流 (test.py)
```
evaluate(cfg, checkpoint_path, logger)
 ├── build_vit(cfg) + load weights
 ├── build_dataloaders(cfg) → val_loader
 └── inference loop (AMP autocast)

predict_single_image(cfg, checkpoint_path, image_path, logger)
 ├── build_vit(cfg) + load weights
 ├── transforms.Compose (Resize→CenterCrop→ToTensor→Normalize)
 └── single inference → top-5 predictions
```

## 关键设计决策

1. **Pre-Norm 结构**: LayerNorm 在 Attention/MLP 之前，训练更稳定
2. **参数分组**: norm/bias/cls_token/pos_embed 不施加 weight_decay
3. **DropPath 线性递增**: 浅层 drop=0，深层逐渐增大到 max_drop_path
4. **Conv2d 实现 Patch Embedding**: 等价于切块+线性投影，但利用 CUDA 卷积核更高效
5. **QKV 合并投影**: 一个 Linear 层同时计算 Q/K/V，减少 kernel launch 次数

## 数据集状态

数据集目录 `dataset/imagenet/` 尚未填充。需要 ImageFolder 格式:
- `train/类别文件夹/图片`
- `val/类别文件夹/图片`

## 环境

- 平台: Windows 10 (WSL2 Ubuntu 24)
- Python 3.10+, CUDA GPU (建议 16GB+)
- Conda 环境名: torch
