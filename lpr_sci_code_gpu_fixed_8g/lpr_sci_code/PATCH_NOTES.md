# Patch Notes

本次修复基于“第3章 TA-RL 主线 + 第4章 KG-RL 主线”的工程实现，重点把 **R-GCN + 知识追踪背景 + 层级奖励** 做成可运行、可复现实验版本。

## 已修复

1. **AMP / PyTorch 2.x 兼容**
   - 修复 `torch.amp.autocast` 调用方式导致的直接崩溃。
2. **padding 与真实概念 ID 冲突**
   - 统一保留 `0` 作为 padding id。
   - 真实概念从 `1..N` 编号。
   - 旧数据集加载时自动修复到新编号空间。
3. **数据不匹配 / shape mismatch**
   - checkpoint 加载支持 `strict=False + resize_mismatched=True`。
   - 增加 `scripts/check_dataset.py` 检查处理后数据。
4. **R-GCN 表现差**
   - 增加显式图预训练 `train_rgcn(...)`，不再只靠下游任务硬学图表示。
5. **平均池化背景 -> 知识追踪背景**
   - 新增 `kb_mode=kt`。
   - 使用 KT 最终隐藏状态 + mastery 概率对历史知识点图嵌入做 mastery-aware 聚合。
6. **终点奖励 -> 层级奖励**
   - 新增 `reward_mode=hierarchical`。
   - 奖励同时考虑：掌握度增益、先修满足、层级进展、复习补救、连贯性、难度匹配、目标命中。
7. **行为克隆阶段不稳定**
   - 强制把真实标签加入候选集，避免 label 被 mask 掉导致 loss 异常。
8. **单卡 8GB 配置**
   - 新增 `configs/junyi_rgcn_kt_hier_8g.yaml`。
   - 提供 `scripts/run_junyi_8g_pipeline.sh` 一键跑。

## 推荐运行顺序

```bash
bash scripts/run_junyi_8g_pipeline.sh ./data/raw/junyi ./data/processed/junyi ./outputs/junyi_rgcn_kt_hier_8g
```

或手动执行：

```bash
python scripts/preprocess_junyi.py --raw_dir ./data/raw/junyi --output_dir ./data/processed/junyi --history_len 20 --path_len 10 --stride 5 --min_user_interactions 25
python scripts/check_dataset.py --dataset_dir ./data/processed/junyi
python scripts/train_kt.py --config configs/junyi_rgcn_kt_hier_8g.yaml --dataset_dir ./data/processed/junyi --output_dir ./outputs/junyi_rgcn_kt_hier_8g
python scripts/train_lpr.py --config configs/junyi_rgcn_kt_hier_8g.yaml --dataset_dir ./data/processed/junyi --kt_ckpt ./outputs/junyi_rgcn_kt_hier_8g/kt_best.pt --output_dir ./outputs/junyi_rgcn_kt_hier_8g
```
