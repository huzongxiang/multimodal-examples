# Part 1 实验整理

## 阶段定位

`Part 1` 是主分支上的最小跨模态对齐阶段。它的主要目的不是直接得到一个高质量的图像描述模型，而是验证以下几点：

- `ViT-Large` 提取出的视觉特征能否通过 `projector` 映射到语言模型的嵌入空间；
- 图像条件驱动的文本生成能否通过因果语言模型损失完成训练；
- 该阶段得到的权重能否作为 `Part 2` 和 `Part 3` 的初始化来源。

因此，这一阶段更适合表述为**形式化的跨模态对齐 / 预热阶段**，而不应直接表述为“高质量 caption 成功训练”。

## 运行配置

- 训练数据：`data\repo\part1_flickr8k\train.jsonl`
- 样本数：`2000`
- 视觉编码器：`google/vit-large-patch16-224`
- 语言模型：`HuggingFaceTB/SmolLM-360M`
- 初始化方式：预训练单模态视觉模型 + 预训练单模态语言模型
- Batch size：`4`
- 学习率：`2e-4`
- 最大长度：`64`
- 完成 epoch 数：`50`
- 全局 step：`25000`

关键产物：

- `output\part1\run_config.json`
- `output\part1\run_summary.json`
- `output\part1\metrics.jsonl`
- `output\part1\losses.json`
- `output\part1\loss_curve.png`
- `output\part1\mini_vlm_full.pt`

## 启动命令示例

以下命令示例默认在项目根目录下执行：

```powershell
$env:PYTHONPATH = "src"
python -m mllm_demo.train `
  --train-jsonl "data\repo\part1_flickr8k\train.jsonl" `
  --output-dir "output\part1" `
  --checkpoint-path "output\part1\training_checkpoint.pt" `
  --vision-model "google/vit-large-patch16-224" `
  --lm-model "HuggingFaceTB/SmolLM-360M" `
  --epochs 50 `
  --batch-size 4 `
  --lr 2e-4 `
  --max-length 64
```

## 训练结果

- 最优 epoch：`49`
- 最优 epoch loss：`0.556580`
- 最终 epoch loss：`0.561619`
- 早停：未触发

从训练流程本身看，这一阶段完成了完整的多模态桥接训练，并生成了可用于后续阶段的初始化权重。

## 小样本评估

评估文件：

- 输入样本：`output\part1\eval_sample_input.jsonl`
- 指标文件：`output\part1\eval_sample\eval_metrics.json`
- 预测文件：`output\part1\eval_sample\predictions.jsonl`

评估设置：

- `12` 条 caption 样本
- 来自当前可用训练集抽样

观测指标：

- Caption exact match：`0.0000`
- Caption token F1：`0.1284`

## 样例输出特征

该阶段的预测往往表现为“语句通顺，但语义偏移明显”。典型现象包括：

- 生成较泛化的场景描述，但与图像内容不匹配；
- 句尾重复；
- 输出发展成长段自然语言，而不是简洁图像描述。

对应样例可见：

- `output\part1\eval_sample\predictions.jsonl`

## 结果解释

这一阶段**作为对齐阶段是成立的**，但**作为独立 caption 模型效果较弱**。

可以成立的结论：

- 多模态连接链路可以训练；
- 图像特征能够进入语言模型并影响生成；
- 该阶段可以为后续任务化微调提供初始化起点。

不宜直接得出的结论：

- `Part 1` 已经是一个效果较好的 caption 模型；
- 该阶段本身已经足以代表强视觉语言能力。

## 后续写作用语建议

建议后续写作中使用如下表述：

- “`Part 1` 用于建立最小视觉—语言接口。”
- “该阶段的主要价值在于跨模态对齐，而不在于最终 caption 性能。”
- “这一阶段可视为资源受限条件下的形式化预训练 / 对齐演示。”
