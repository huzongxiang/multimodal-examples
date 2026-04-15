# Repo Alignment

这份说明只关注你在意的三件事：模型、数据来源、训练阶段。

## Part 1

- 数据源：`jxie/flickr8k`
- 本地数据：`D:\codex\mllm-demo\data\repo\part1_flickr8k\train.jsonl`
- 样本数：`2000`
- 模型：
  - vision: `google/vit-large-patch16-224`
  - lm: `HuggingFaceTB/SmolLM-360M`
- 超参：
  - epochs: `6`
  - batch size: `4`
  - lr: `2e-4`
  - max length: `64`
- 训练脚本：`D:\codex\mllm-demo\scripts\train_part1_like_repo.ps1`

## Part 2

- 数据源：`Francesco/animals-ij5d2`
- 本地数据：`D:\codex\mllm-demo\data\repo\part2_animals_od\train.jsonl`
- 样本数：`800`
- 初始化：从 `Part 1` 的 caption checkpoint 继续训练
- 模型：
  - vision: `google/vit-large-patch16-224`
  - lm: `HuggingFaceTB/SmolLM-360M`
- 超参：
  - epochs: `10`
  - batch size: `4`
  - lr: `1e-4`
  - max length: `256`
- 训练脚本：`D:\codex\mllm-demo\scripts\train_part2_like_repo.ps1`

## Part 3

- 数据源：`lmms-lab/VQAv2`
- 本地数据：
  - train: `D:\codex\mllm-demo\data\repo\part3_vqav2\train.jsonl`
  - test: `D:\codex\mllm-demo\data\repo\part3_vqav2\test.jsonl`
- 样本数：
  - train: `3600`
  - test: `400`
- 初始化：从 `Part 1` 的 caption checkpoint 继续训练
- 模型：
  - vision: `google/vit-large-patch16-224`
  - lm: `HuggingFaceTB/SmolLM-360M`
- 超参：
  - epochs: `12`
  - batch size: `8`
  - lr: `2e-4`
  - max length: `64`
- 训练脚本：`D:\codex\mllm-demo\scripts\train_part3_like_repo.ps1`

## Part 4

- 数据源：
  - caption: `jxie/flickr8k`，`1000`
  - detection: `Francesco/animals-ij5d2`，`400`
  - vqa: `lmms-lab/VQAv2`，`1000`
- 本地数据：`D:\codex\mllm-demo\data\repo\part4_multitask\train.jsonl`
- 总样本数：`2400`
- 初始化：从一个 `base + 135M` 的 caption checkpoint 继续训练
- 模型：
  - vision: `google/vit-base-patch16-224`
  - lm: `HuggingFaceTB/SmolLM-135M`
- 超参：
  - epochs: `8`
  - batch size: `4`
  - lr: `1e-4`
  - max length: `256`
- 训练脚本：
  - 先跑 caption bootstrap：`D:\codex\mllm-demo\scripts\train_part1_for_part4_branch.ps1`
  - 再跑 multitask：`D:\codex\mllm-demo\scripts\train_part4_like_repo.ps1`

## 说明

- `Part 1/2/3` 和 `Part 4` 在项目公开页面上不是同一套模型规模。
- 所以我把它拆成了两条复现分支：
  - 大模型分支：`Part 1 -> Part 2 -> Part 3`
  - 小模型分支：`Part 1 for Part 4 -> Part 4`
- 数据存储格式用了本地 JSONL，只是为了方便跑脚本，不影响与原项目的数据源和阶段设置对齐。
