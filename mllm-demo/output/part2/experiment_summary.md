# Part 2 实验整理

## 阶段定位

`Part 2` 的任务是将目标检测改写为文本生成问题。该阶段的重点不是追求专门检测器级别的性能，而是验证：

`图像 + 提示词 -> JSON 文本`

这一统一生成接口是否能够承载结构化视觉任务。

本阶段从 `Part 1` 的权重继续训练，用来检验共享的多模态接口是否能够支持结构化输出。

## 运行配置

- 训练数据：`data\repo\part2_animals_od\train\data.jsonl`
- 小样本评估来源：`data\repo\part2_animals_od\test\data.jsonl`
- 训练样本数：`800`
- 测试样本数：`200`
- 视觉编码器：`google/vit-large-patch16-224`
- 语言模型：`HuggingFaceTB/SmolLM-360M`
- 初始化来源：`output\part1`
- Batch size：`4`
- 学习率：`1e-4`
- 最大长度：`256`
- 完成 epoch 数：`49`
- 全局 step：`9800`
- 早停 epoch：`49`

关键产物：

- `output\part2\run_config.json`
- `output\part2\run_summary.json`
- `output\part2\metrics.jsonl`
- `output\part2\losses.json`
- `output\part2\loss_curve.png`
- `output\part2\mini_vlm_full.pt`

## 启动命令示例

以下命令示例默认在项目根目录下执行：

```powershell
$env:PYTHONPATH = "src"
python -m mllm_demo.train `
  --train-jsonl "data\repo\part2_animals_od\train\data.jsonl" `
  --output-dir "output\part2" `
  --checkpoint-path "output\part2\training_checkpoint.pt" `
  --init-from "output\part1" `
  --vision-model "google/vit-large-patch16-224" `
  --lm-model "HuggingFaceTB/SmolLM-360M" `
  --epochs 50 `
  --batch-size 4 `
  --lr 1e-4 `
  --max-length 256 `
  --early-stop-patience 5 `
  --early-stop-min-delta 0.003
```

## 训练结果

- 最优 epoch：`44`
- 最优 epoch loss：`0.004608`
- 最终 epoch loss：`0.006064`
- 早停：触发

与 `Part 1` 相比，这一阶段收敛得更稳定，这与检测任务输出空间更受约束有关。

## 小样本评估

评估文件：

- 输入样本：`output\part2\eval_sample_input.jsonl`
- 指标文件：`output\part2\eval_sample\eval_metrics.json`
- 预测文件：`output\part2\eval_sample\predictions.jsonl`

评估设置：

- `12` 条目标检测样本
- 从测试集中抽样

观测指标：

- JSON parse success：`1.0000`
- Object count accuracy：`0.9167`
- Label accuracy：`0.7917`
- Mean bbox IoU：`0.4078`

## 样例输出特征

从样例可以看到，模型通常能够：

- 保持合法 JSON 结构；
- 较高概率给出正确类别；
- 生成可用但不完全精确的框。

可参考：

- `output\part2\eval_sample\predictions.jsonl`

典型现象包括：

- 类别预测正确但框位置较粗；
- 单目标图像预测较稳定；
- 多目标图像中偶尔会漏计目标数量。

## 结果解释

`Part 2` 可以视为**明确有效的阶段**。

说明了什么：

- 目标检测可以被改写成文本生成问题；
- `Part 1` 建立的多模态桥接已经足以支持结构化视觉任务；
- 统一的因果语言模型训练路径可以输出机器可解析的检测结果。

局限：

- 框位置精度仍明显弱于专门检测器；
- 目标数量并非总是准确。

## 后续写作用语建议

建议后续写作中使用如下表述：

- “`Part 2` 说明结构化视觉预测可以通过统一生成接口实现。”
- “模型在这一阶段不再只是描述图像，而是开始表达图像中的结构信息。”
