# LPR-SCI 实验代码（对应“第4章最省事、成功率最高版本”）

> 2026-04 修复说明：本版额外修复了 **AMP 兼容问题、padding 与真实概念 ID 冲突、R-GCN 无显式图预训练、行为克隆标签不在候选集、KT/LPR 维度不一致自适应加载、配置路径解析不稳** 等常见崩溃点，并补上了 **R-GCN + 知识追踪背景 + 层级奖励** 的可运行版本。

这套代码把你原第4章的主线保留下来，并补成一套能直接做论文实验的工程：

- 保留：**时间感知注意力 + 强化学习逐步生成路径**
- 原版复现开关：**TransE + 平均池化 + 稀疏终点奖励 + 经典认知导航**
- 改进版开关：**多关系 R-GCN + 注意力聚合（已去掉遗忘建模） + 稠密层级奖励 + 复习增强候选集**

## 1. 代码能做什么

### 1.1 已实现模型

- **原版 KG-RL 近似复现**：`graph_type=transe`, `kb_mode=mean`, `reward_mode=sparse`, `candidate_mode=classic`
- **保守升级版（推荐投稿版）**：`graph_type=rgcn`, `kb_mode=kt`, `reward_mode=hierarchical`, `candidate_mode=review_augmented`
- **单卡 8GB 推荐配置**：`configs/junyi_rgcn_kt_hier_8g.yaml`
- **兼容别名配置**：`configs/junyi_rgcn_kt_hier_stage2.yaml`
- **TA-RL 风格消融**：`kb_mode=zero`
- 简单基线：`random`, `popularity`, `seqknn`, `gru4rec`

### 1.2 已实现实验流程

1. 数据预处理（JunYi / Eedi）
2. 检查处理后数据是否存在 ID / padding / 图结构错位（`check_dataset.py`）
3. 训练 KT 环境模拟器（`train_kt.py`）
4. 训练学习路径模型：先 **行为克隆**，再 **RL 微调**（`train_lpr.py`）
5. 运行简单基线（`run_baselines.py`）
6. 单独评估与导出预测路径（`evaluate_lpr.py`）
7. 一键跑 JunYi 单卡 8GB 管线（`run_junyi_8g_pipeline.sh`）

### 1.3 当前指标

- `mastery_gain`
- `hit_rate`
- `ndcg@10`
- `mrr`
- `prereq_violation`
- `difficulty_smoothness`
- `review_coverage`

这些指标已经足够支撑一版 SCI 初稿。

## 2. 安装

在项目根目录执行：

```bash
pip install -e .
```

如果不想 editable install，也可以：

```bash
pip install -r requirements.txt
export PYTHONPATH=$PWD/src
```

## 3. 先跑通 toy 数据（强烈建议）

```bash
python examples/make_toy_data.py --output_dir ./toy_processed
python scripts/train_kt.py --config configs/toy_full.yaml --dataset_dir ./toy_processed --output_dir ./outputs/toy
python scripts/train_lpr.py --config configs/toy_full.yaml --dataset_dir ./toy_processed --kt_ckpt ./outputs/toy/kt_best.pt --output_dir ./outputs/toy
python scripts/run_baselines.py --config configs/toy_full.yaml --dataset_dir ./toy_processed --kt_ckpt ./outputs/toy/kt_best.pt --output_dir ./outputs/toy/baselines --include_gru4rec
```

## 4. JunYi 推荐命令

现在 `configs/junyi_rgcn_kt_hier_8g.yaml` 和 `configs/junyi_rgcn_kt_hier_stage2.yaml` 都可用，内容保持一致。

```bash
python scripts/train_kt.py --config configs/junyi_rgcn_kt_hier_8g.yaml --dataset_dir ./data/processed/junyi --output_dir ./outputs/junyi_rgcn_kt_hier_8g
python scripts/train_lpr.py --config configs/junyi_rgcn_kt_hier_8g.yaml --dataset_dir ./data/processed/junyi --kt_ckpt ./outputs/junyi_rgcn_kt_hier_8g/kt_best.pt --output_dir ./outputs/junyi_rgcn_kt_hier_8g
```

## 5. 路径解析说明

现在训练、评估和基线脚本会优先解析你传入的命令行路径；当配置文件里使用相对路径时，会自动兼容当前工作目录和项目根目录。

这意味着即使你不是在仓库根目录下启动脚本，也不容易再因为 `configs/...yaml`、`data/...`、`outputs/...` 这类相对路径而报错。
