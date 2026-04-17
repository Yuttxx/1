# LPR-SCI 实验代码（对应“第4章最省事、成功率最高版本”）

> 2026-04 修复说明：本版额外修复了 **AMP 兼容问题、padding 与真实概念 ID 冲突、R-GCN 无显式图预训练、行为克隆标签不在候选集、KT/LPR 维度不一致自适应加载** 等常见崩溃点，并补上了 **R-GCN + 知识追踪背景 + 层级奖励** 的可运行版本。

这套代码把你原第4章的主线保留下来，并补成一套能直接做论文实验的工程：

- 保留：**时间感知注意力 + 强化学习逐步生成路径**
- 原版复现开关：**TransE + 平均池化 + 稀疏终点奖励 + 经典认知导航**
- 改进版开关：**多关系 R-GCN + 注意力聚合（已去掉遗忘建模） + 稠密层级奖励 + 复习增强候选集**

## 1. 代码能做什么

### 1.1 已实现模型

- **原版 KG-RL 近似复现**：`graph_type=transe`, `kb_mode=mean`, `reward_mode=sparse`, `candidate_mode=classic`
- **保守升级版（推荐投稿版）**：`graph_type=rgcn`, `kb_mode=kt`, `reward_mode=hierarchical`, `candidate_mode=review_augmented`
- **单卡 8GB 推荐配置**：`configs/junyi_rgcn_kt_hier_8g.yaml`
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

---

## 2. 目录结构

```text
lpr_sci_code/
├── configs/
│   ├── junyi_sci_conservative.yaml
│   ├── eedi_sci_conservative.yaml
│   ├── junyi_original_kg_rl.yaml
│   ├── junyi_ta_ablation.yaml
│   └── toy_full.yaml
├── examples/
│   └── make_toy_data.py
├── scripts/
│   ├── preprocess_junyi.py
│   ├── preprocess_eedi.py
│   ├── train_kt.py
│   ├── train_lpr.py
│   ├── evaluate_lpr.py
│   └── run_baselines.py
├── src/lpr/
│   ├── baselines.py
│   ├── common.py
│   ├── data.py
│   ├── graph_utils.py
│   ├── metrics.py
│   ├── models.py
│   ├── rl.py
│   └── trainers.py
├── pyproject.toml
└── requirements.txt
```

---

## 3. 安装

在项目根目录执行：

```bash
pip install -e .
```

如果不想 editable install，也可以：

```bash
pip install -r requirements.txt
export PYTHONPATH=$PWD/src
```

---

## 4. 先跑通 toy 数据（强烈建议）

```bash
python examples/make_toy_data.py --output_dir ./toy_processed
python scripts/train_kt.py --config configs/toy_full.yaml --dataset_dir ./toy_processed --output_dir ./outputs/toy
python scripts/train_lpr.py --config configs/toy_full.yaml --dataset_dir ./toy_processed --kt_ckpt ./outputs/toy/kt_best.pt --output_dir ./outputs/toy
python scripts/run_baselines.py --config configs/toy_full.yaml --dataset_dir ./toy_processed --kt_ckpt ./outputs/toy/kt_best.pt --output_dir ./outputs/toy/baselines --include_gru4rec
```

先在 toy 数据上把整个链路跑通，再换 JunYi / Eedi。

---

## 5. JunYi 实验怎么跑

### 5.1 原始文件准备

把 JunYi 原始文件放到例如 `./data/raw/junyi/`：

```text
./data/raw/junyi/
├── junyi_ProblemLog_original.csv
├── junyi_Exercise_table.csv
├── relationship_annotation_training.csv        # 可选但强烈建议
└── relationship_annotation_testing.csv         # 可选但强烈建议
```

### 5.2 预处理

```bash
python scripts/preprocess_junyi.py \
  --raw_dir ./data/raw/junyi \
  --output_dir ./data/processed/junyi \
  --history_len 20 \
  --path_len 10 \
  --stride 5 \
  --min_user_interactions 25
```

如果你机器内存不够，先用快速版本：

```bash
python scripts/preprocess_junyi.py \
  --raw_dir ./data/raw/junyi \
  --output_dir ./data/processed/junyi_small \
  --history_len 20 \
  --path_len 10 \
  --stride 5 \
  --min_user_interactions 25 \
  --max_users 5000
```

### 5.2.1 检查处理后的数据

```bash
python scripts/check_dataset.py --dataset_dir ./data/processed/junyi
```

### 5.3 训练 KT 模拟器

```bash
python scripts/train_kt.py \
  --config configs/junyi_rgcn_kt_hier_8g.yaml \
  --dataset_dir ./data/processed/junyi \
  --output_dir ./outputs/junyi_rgcn_kt_hier_8g
```

### 5.4 训练原版 KG-RL

```bash
python scripts/train_lpr.py \
  --config configs/junyi_original_kg_rl.yaml \
  --dataset_dir ./data/processed/junyi \
  --kt_ckpt ./outputs/junyi_sci_conservative/kt_best.pt \
  --output_dir ./outputs/junyi_original_kgrl
```

### 5.5 训练改进版（推荐）

```bash
python scripts/train_lpr.py \
  --config configs/junyi_rgcn_kt_hier_8g.yaml \
  --dataset_dir ./data/processed/junyi \
  --kt_ckpt ./outputs/junyi_rgcn_kt_hier_8g/kt_best.pt \
  --output_dir ./outputs/junyi_rgcn_kt_hier_8g
```

### 5.6 跑简单基线

```bash
python scripts/run_baselines.py \
  --config configs/junyi_rgcn_kt_hier_8g.yaml \
  --dataset_dir ./data/processed/junyi \
  --kt_ckpt ./outputs/junyi_rgcn_kt_hier_8g/kt_best.pt \
  --output_dir ./outputs/junyi_baselines \
  --include_gru4rec
```

---

## 6. Eedi 实验怎么跑

### 6.1 原始文件准备

把 Eedi Task 3/4 所需文件放到例如 `./data/raw/eedi/`：

```text
./data/raw/eedi/
├── train_task_3_4.csv
├── question_metadata_task_3_4.csv
└── subject_metadata.csv
```

### 6.2 预处理

```bash
python scripts/preprocess_eedi.py \
  --raw_dir ./data/raw/eedi \
  --output_dir ./data/processed/eedi \
  --history_len 20 \
  --path_len 10 \
  --stride 5 \
  --min_user_interactions 20
```

### 6.3 训练与评估

```bash
python scripts/train_kt.py \
  --config configs/eedi_sci_conservative.yaml \
  --dataset_dir ./data/processed/eedi \
  --output_dir ./outputs/eedi_sci_conservative

python scripts/train_lpr.py \
  --config configs/eedi_sci_conservative.yaml \
  --dataset_dir ./data/processed/eedi \
  --kt_ckpt ./outputs/eedi_sci_conservative/kt_best.pt \
  --output_dir ./outputs/eedi_sci_conservative
```

---

## 7. 你应该怎么做论文表格

### 7.1 主结果表

至少报告：

- random
- popularity
- seqknn
- gru4rec
- 原版 KG-RL（`junyi_original_kg_rl.yaml`）
- TA 风格消融（`junyi_ta_ablation.yaml`）
- 改进版（`junyi_rgcn_kt_hier_8g.yaml`）

### 7.2 消融表

依次去掉：

- `R-GCN -> TransE`
- `kt -> mean`
- `hierarchical reward -> sparse reward`
- `review_augmented -> classic`

### 7.3 额外分析

建议另外做三类图：

- 路径长度变化（5 / 10 / 15）
- 冷启动用户 vs 活跃用户
- 不同目标难度分组

---

## 8. 这份代码和你原第4章的对应关系

### 原第4章 4.3.1 时间感知注意力
对应：

- `src/lpr/models.py -> TimeAwarePreferenceEncoder`

### 原第4章 4.3.2 TransE 知识背景
对应：

- 原版：`TransEEncoder + kb_mode=mean`
- 改进版：`RGCNEncoder + kb_mode=forget_attention`

### 原第4章 4.3.3 认知导航候选集
对应：

- `src/lpr/rl.py -> CandidateGenerator`

### 原第4章 4.3.4 强化学习推荐
对应：

- `src/lpr/trainers.py -> LPRTrainer`
- 先 BC，再 RL 微调

---

## 9. 你接下来最该改哪几个地方

### 第一优先级
直接跑：

- `junyi_original_kg_rl.yaml`
- `junyi_sci_conservative.yaml`

把两者在 JunYi 上的结果先做出来。

### 第二优先级
再跑 Eedi：

- `eedi_sci_conservative.yaml`

只要你在两个数据集上都能稳定优于原版 KG-RL，论文主体就立住了。

### 第三优先级
把 `run_baselines.py` 结果和你的方法结果放成总表，再补案例分析。

---

## 10. 这份代码里我替你做好的研究假设

为了让工程尽快可跑，我做了这些明确设定：

1. **JunYi**：练习名直接视为 concept id。
2. **Eedi**：一道题如果对应多个 subject，默认取“层级最深”的 subject 作为主 concept。
3. **RL 环境**：使用训练好的 KT 模型充当学生模拟器。
4. **改进版奖励**：不是只看终点命中，而是同时考虑掌握提升、先修合理性、难度匹配、复习收益和重复惩罚。
5. **最省事版本**：优先保证代码闭环与实验可落地，不追求最复杂的 SOTA 结构。

---

## 11. 你写论文时必须诚实说明的限制

这套代码已经足够支撑 SCI 初稿，但有两个地方你在论文里必须写清楚：

1. **环境评估依赖 KT 模拟器**，因此这是 offline / simulator-based evaluation，不等价于真实在线教学增益。
2. **Eedi 主 concept 映射是工程化简化**，后续可以替换成多概念版本。

---

## 12. 最后给你的最短执行路径

如果你现在只想最快出结果，按下面顺序：

```bash
# 1) JunYi 预处理
python scripts/preprocess_junyi.py --raw_dir ./data/raw/junyi --output_dir ./data/processed/junyi

# 2) KT 模拟器
python scripts/train_kt.py --config configs/junyi_sci_conservative.yaml --dataset_dir ./data/processed/junyi --output_dir ./outputs/junyi_sci_conservative

# 3) 原版 KG-RL
python scripts/train_lpr.py --config configs/junyi_original_kg_rl.yaml --dataset_dir ./data/processed/junyi --kt_ckpt ./outputs/junyi_sci_conservative/kt_best.pt --output_dir ./outputs/junyi_original_kgrl

# 4) 改进版
python scripts/train_lpr.py --config configs/junyi_sci_conservative.yaml --dataset_dir ./data/processed/junyi --kt_ckpt ./outputs/junyi_sci_conservative/kt_best.pt --output_dir ./outputs/junyi_sci_conservative

# 5) 简单基线
python scripts/run_baselines.py --config configs/junyi_sci_conservative.yaml --dataset_dir ./data/processed/junyi --kt_ckpt ./outputs/junyi_sci_conservative/kt_best.pt --output_dir ./outputs/junyi_baselines --include_gru4rec
```

你先把这 5 步跑完，再开始补 Eedi。


## 9. 单卡 8GB / 内存 32GB 推荐配置

直接用这个配置：

```bash
python scripts/train_kt.py \
  --config configs/junyi_single_gpu.yaml \
  --dataset_dir ./data/processed/junyi \
  --output_dir ./outputs/junyi_single_gpu

python scripts/train_lpr.py \
  --config configs/junyi_single_gpu.yaml \
  --dataset_dir ./data/processed/junyi \
  --kt_ckpt ./outputs/junyi_single_gpu/kt_best.pt \
  --output_dir ./outputs/junyi_single_gpu
```

如果你已经预处理过 JunYi，新的 `load_standard_dataset()` 会自动修复旧版数据里可能出现的 concept id 不连续问题；如果你准备重新预处理，`preprocess_junyi.py` 也已经修复了重复概念名导致的 id 错位问题。
