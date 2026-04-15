# Part 4 实验整理

## 阶段定位

`Part 4` 是整个教学 demo 的最终多任务阶段。它要检验的是：同一个小型多模态模型能否在统一生成接口下，在以下任务之间切换：

- caption；
- 以 JSON 文本形式表达的 object detection；
- visual question answering。

这是当前实验中最能体现“最小 MLLM”形态的阶段。

## 运行配置

- 训练数据：`data\repo\part4_multitask\train.jsonl`
- 数据组成：
  - caption：`1000`
  - object_detection：`400`
  - vqa：`1000`
  - 合计：`2400`
- 视觉编码器：`google/vit-base-patch16-224`
- 语言模型：`HuggingFaceTB/SmolLM-135M`
- 初始化来源：`output\part1_for_part4`
- Batch size：`4`
- 学习率：`1e-4`
- 最大长度：`256`
- 完成 epoch 数：`50`
- 全局 step：`30000`

关键产物：

- `output\part4\run_config.json`
- `output\part4\run_summary.json`
- `output\part4\metrics.jsonl`
- `output\part4\losses.json`
- `output\part4\loss_curve.png`
- `output\part4\mini_vlm_full.pt`

## 启动命令示例

以下命令示例默认在项目根目录下执行：

```powershell
$env:PYTHONPATH = "src"
python -m mllm_demo.train `
  --train-jsonl "data\repo\part4_multitask\train.jsonl" `
  --output-dir "output\part4" `
  --checkpoint-path "output\part4\training_checkpoint.pt" `
  --init-from "output\part1_for_part4" `
  --vision-model "google/vit-base-patch16-224" `
  --lm-model "HuggingFaceTB/SmolLM-135M" `
  --epochs 50 `
  --batch-size 4 `
  --lr 1e-4 `
  --max-length 256 `
  --early-stop-patience 5 `
  --early-stop-min-delta 0.005
```

## 训练结果

- 最优 epoch：`47`
- 最优 epoch loss：`0.333559`
- 最终 epoch loss：`0.336523`
- 早停：未触发

该阶段完整跑满设定 epoch，并生成了稳定的多任务 checkpoint。

## 平衡小样本评估

评估文件：

- 输入样本：`output\part4\eval_sample_input.jsonl`
- 指标文件：`output\part4\eval_sample\eval_metrics.json`
- 预测文件：`output\part4\eval_sample\predictions.jsonl`

评估设置：

- `18` 条平衡样本
  - `6` 条 caption
  - `6` 条 object_detection
  - `6` 条 vqa
- 从训练混合集合中抽样

观测指标：

- Caption exact match：`0.1667`
- Caption token F1：`0.7964`
- Detection JSON parse success：`1.0000`
- Detection object count accuracy：`0.8333`
- Detection label accuracy：`1.0000`
- Detection bbox IoU：`0.8595`
- VQA exact match：`1.0000`
- VQA token F1：`1.0000`

评估注意事项：

- VQA 指标使用 `prediction_content`，即先剥离 prompt 前缀后再评分；
- 否则像 `Question ... Answer: birthday` 这类正确回答会被低估。

## 样例输出特征

样例输出已经明显体现出任务切换能力：

- Caption：
  - “Two children are riding horses over sand near a white fence.”
  - “A black and white dog runs on the grass.”
- Detection：
  - 对部分 racoon、goat、dog 样本可恢复与目标几乎一致的 JSON；
  - 在拥挤场景中会出现漏计目标，但标签仍准确。
- VQA：
  - `"birthday"`、`"3"`、`"shadow"`、`"plane"`、`"cheese and oregano"`、`"no"` 等样本均回答正确。

详见：

- `output\part4\eval_sample\predictions.jsonl`

## 结果解释

`Part 4` 是当前实验中最有说服力的阶段。

说明了什么：

- 同一套多模态骨架可以服务于多种视觉语言任务；
- prompt 模板和目标文本格式足以驱动任务切换；
- 当前教学 demo 中，“最小 MLLM”的核心形态主要由这一阶段体现出来。

局限：

- 当前平衡评估集来自训练混合数据抽样，因此更适合说明“模型已经学会任务切换与统一接口”，而不宜直接表述为严格泛化能力验证；
- 因此最稳的结论应落在**统一接口能力和多任务行为成立**，而不是 benchmark 性能。

## 后续写作用语建议

建议后续写作中使用如下表述：

- “`Part 4` 说明，一旦视觉任务被重写为统一生成接口，同一个小型多模态模型便能够通过 prompt 在不同任务之间切换行为。”
- “该阶段最能体现本教学案例中最小 MLLM 的核心思想。”
