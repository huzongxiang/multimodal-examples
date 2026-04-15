# Output Layout

所有正式训练产物统一输出到项目内 `output/`。

## 目录约定

- `D:\codex\mllm-demo\output\part1`
- `D:\codex\mllm-demo\output\part2`
- `D:\codex\mllm-demo\output\part3`
- `D:\codex\mllm-demo\output\part1_for_part4`
- `D:\codex\mllm-demo\output\part4`

## 每个阶段目录中的文件

- `mini_vlm_full.pt`
- `training_checkpoint.pt`
- `metrics.jsonl`
- `losses.json`
- `eval_epoch_XXX.json`
- `eval_predictions_epoch_XXX.jsonl`
- `loss_curve.png`

## 说明

- 后续写教学案例文档时，只从 `output/` 目录取结果。
- `checkpoints/` 是之前错误使用的目录，不再作为正式输出目录。
