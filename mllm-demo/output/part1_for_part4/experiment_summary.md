# Part 1 for Part 4 实验整理

## 阶段定位

`Part 1 for Part 4` 是为多任务小模型分支单独准备的 caption 预热阶段。设置这一阶段的原因是 `Part 4` 使用的是另一组更小的模型规格：

- `ViT-Base`
- `SmolLM-135M`

因此，这一阶段更适合被理解为**面向 `Part 4` 的小模型分支对齐预热**，而不是单独的主结论阶段。

## 运行配置

- 训练数据：`data\repo\part1_flickr8k\train.jsonl`
- 小样本评估来源：从训练集抽样
- 样本数：`2000`
- 视觉编码器：`google/vit-base-patch16-224`
- 语言模型：`HuggingFaceTB/SmolLM-135M`
- 初始化方式：预训练单模态视觉模型 + 预训练单模态语言模型
- Batch size：`4`
- 学习率：`2e-4`
- 最大长度：`64`
- 完成 epoch 数：`50`
- 全局 step：`25000`

关键产物：

- `output\part1_for_part4\run_config.json`
- `output\part1_for_part4\run_summary.json`
- `output\part1_for_part4\metrics.jsonl`
- `output\part1_for_part4\losses.json`
- `output\part1_for_part4\loss_curve.png`
- `output\part1_for_part4\mini_vlm_full.pt`

## 启动命令示例

以下命令示例默认在项目根目录下执行：

```powershell
$env:PYTHONPATH = "src"
python -m mllm_demo.train `
  --train-jsonl "data\repo\part1_flickr8k\train.jsonl" `
  --output-dir "output\part1_for_part4" `
  --checkpoint-path "output\part1_for_part4\training_checkpoint.pt" `
  --vision-model "google/vit-base-patch16-224" `
  --lm-model "HuggingFaceTB/SmolLM-135M" `
  --epochs 50 `
  --batch-size 4 `
  --lr 2e-4 `
  --max-length 64 `
  --early-stop-patience 5 `
  --early-stop-min-delta 0.005
```

## 训练结果

- 最优 epoch：`49`
- 最优 epoch loss：`0.987185`
- 最终 epoch loss：`0.988868`
- 早停：未触发

与主分支 `Part 1` 相比，小模型分支最终 loss 明显更高。

## 小样本评估

评估文件：

- 输入样本：`output\part1_for_part4\eval_sample_input.jsonl`
- 指标文件：`output\part1_for_part4\eval_sample\eval_metrics.json`
- 预测文件：`output\part1_for_part4\eval_sample\predictions.jsonl`

评估设置：

- `12` 条 caption 样本
- 从训练集中抽样

观测指标：

- Caption exact match：`0.0000`
- Caption token F1：`0.1873`

## 样例输出特征

该分支作为自由 caption 模型仍然较弱，但相对于主分支 `Part 1`，输出更像简短场景描述。典型表现包括：

- 生成较泛化的场景句子；
- 句尾重复；
- 有一定语义重合，但细节仍明显不足。

对应样例见：

- `output\part1_for_part4\eval_sample\predictions.jsonl`

## 结果解释

这一阶段更适合表述为**对 `Part 4` 分支有效的初始化准备**。

说明了什么：

- 小一档的视觉 backbone 和语言 backbone 同样可以通过 projector 方式建立对齐；
- 得到的 checkpoint 可以作为多任务训练的起点。

不宜直接声称：

- 这一阶段已经形成较强的独立 caption 能力。

## 后续写作用语建议

建议后续写作中使用如下表述：

- “`Part 1 for Part 4` 是为小模型多任务分支准备的对齐预热阶段。”
- “其主要作用是提供初始化，而不是单独追求 caption 性能。”
