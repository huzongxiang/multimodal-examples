# Part 3 实验整理

## 阶段定位

`Part 3` 引入视觉问答任务。该阶段的目标是检验：同一个多模态骨架是否能够在图像条件下，对文本问题生成回答，而不只是做 caption 或结构化检测输出。

核心任务形式为：

`图像 + Question: ... Answer: -> 答案文本`

## 运行配置

- 训练数据：`data\repo\part3_vqav2\train.jsonl`
- 小样本评估来源：`data\repo\part3_vqav2\test.jsonl`
- 训练样本数：`3600`
- 测试样本数：`400`
- 视觉编码器：`google/vit-large-patch16-224`
- 语言模型：`HuggingFaceTB/SmolLM-360M`
- 初始化来源：`output\part1`
- Batch size：`8`
- 学习率：`2e-4`
- 最大长度：`64`
- 完成 epoch 数：`14`
- 全局 step：`5400`
- 早停 epoch：`14`

关键产物：

- `output\part3\run_config.json`
- `output\part3\run_summary.json`
- `output\part3\metrics.jsonl`
- `output\part3\losses.json`
- `output\part3\loss_curve.png`
- `output\part3\mini_vlm_full.pt`

## 启动命令示例

以下命令示例默认在项目根目录下执行：

```powershell
$env:PYTHONPATH = "src"
python -m mllm_demo.train `
  --train-jsonl "data\repo\part3_vqav2\train.jsonl" `
  --output-dir "output\part3" `
  --checkpoint-path "output\part3\training_checkpoint.pt" `
  --init-from "output\part1" `
  --vision-model "google/vit-large-patch16-224" `
  --lm-model "HuggingFaceTB/SmolLM-360M" `
  --epochs 50 `
  --batch-size 8 `
  --lr 2e-4 `
  --max-length 64 `
  --early-stop-patience 5 `
  --early-stop-min-delta 0.005
```

## 训练结果

- 最优 epoch：`9`
- 最优监控值：`0.004329`
- 最优 epoch loss：`0.000973`
- 最终 epoch loss：`0.001156`
- 早停：在第 `14` 轮触发

训练过程中需要对画图逻辑做稳定性修正，最终训练正常完成并生成了完整结果目录。

## 小样本评估

评估文件：

- 输入样本：`output\part3\eval_sample_input.jsonl`
- 指标文件：`output\part3\eval_sample\eval_metrics.json`
- 预测文件：`output\part3\eval_sample\predictions.jsonl`

评估设置：

- `12` 条 VQA 样本
- 从测试集中抽样

观测指标：

- Exact match：`0.3333`
- Token F1：`0.3333`

评估注意事项：

- 当前指标使用 `prediction_content`，即先去掉 prompt 前缀再算分；
- 否则像 `Question ... Answer: birthday` 这种本质答对的输出，会被误判为错误。

## 样例输出特征

该阶段已经能回答一部分短答案问题，尤其是：

- yes/no 问题；
- 简短的具体属性或类别问题。

但仍有明显错误，例如：

- 计数错误；
- 给出看似合理但不正确的属性；
- 只抓住部分视觉线索，导致答案偏差。

例如：

- 正确：`Is the dog chasing the sheep? -> yes`
- 错误：`How many trees? -> 3`，而参考答案为 `6`
- 错误：`What color are the batter's sneakers? -> white`，而参考答案为 `red`

对应样例见：

- `output\part3\eval_sample\predictions.jsonl`

## 结果解释

`Part 3` 可以视为**部分有效**。

说明了什么：

- 模型已经从“图像到文本生成”推进到“图像 + 文本条件到答案生成”；
- 统一多模态接口已经能支持基础 VQA 行为；
- 系统开始表现出较明确的 instruction-following 形态。

局限：

- 回答准确率仍有限；
- 相比结构更受约束的 detection 任务，VQA 难度更高。

## 后续写作用语建议

建议后续写作中使用如下表述：

- “`Part 3` 展示了模型从图像描述走向条件性回答的过程。”
- “该阶段已经具备基础视觉问答能力，但精度仍然有限。”
