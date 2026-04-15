# MLLM Demo

如果你要按 `vlm-from-scratch` 的项目配置复现，优先看 [REPO_ALIGNMENT.md](D:/codex/mllm-demo/REPO_ALIGNMENT.md)。这个文件记录了当前已经对齐好的模型名、数据来源、样本数和分阶段训练脚本。

现在训练过程会额外记录：

- `metrics.jsonl`：逐 step / 逐 epoch 的 loss 日志
- `losses.json`：epoch 平均 loss
- `eval_epoch_XXX.json`：训练期可选评估结果
- `eval_predictions_epoch_XXX.jsonl`：训练期样例预测
- `mini_vlm_full.pt`：最终模型

训练后也可以单独跑评估和曲线：

```powershell
powershell -ExecutionPolicy Bypass -File D:\codex\mllm-demo\scripts\eval_checkpoint.ps1 `
  -CheckpointDir D:\codex\mllm-demo\checkpoints\mini-vlm-flickr8k `
  -EvalJsonl D:\codex\mllm-demo\data\repo\part1_flickr8k\train.jsonl `
  -OutputDir D:\codex\mllm-demo\checkpoints\mini-vlm-flickr8k\eval
```

```powershell
powershell -ExecutionPolicy Bypass -File D:\codex\mllm-demo\scripts\plot_metrics.ps1 `
  -MetricsPath D:\codex\mllm-demo\checkpoints\mini-vlm-flickr8k\metrics.jsonl `
  -OutputPath D:\codex\mllm-demo\checkpoints\mini-vlm-flickr8k\loss_curve.png
```

这是一个按 `vlm-from-scratch` 思路整理出来的本地可跑版教学 demo。它不是从随机初始化预训练完整 MLLM，而是用预训练视觉编码器和预训练语言模型，自己实现一个最小多模态桥接层，再按阶段做任务训练：

1. `caption`：先学看图说话
2. `detection`：把目标检测改写成 JSON 文本生成
3. `vqa`：把图像问答改写成指令跟随生成
4. `multitask`：把三种任务混合训练

## 目录

```text
mllm-demo/
  data/
    tiny/
      images/
      caption.jsonl
      detection.jsonl
      vqa.jsonl
      multitask.jsonl
  src/mllm_demo/
    data.py
    infer.py
    model.py
    train.py
  tools/
    build_tiny_dataset.py
```

## 当前可直接用的 Python

本机已经发现一个可用解释器：

```powershell
D:\codex\multimodal-examples\.conda-envs\tld-demo\python.exe
```

下面命令都默认用它。

## 1. 先生成 tiny 数据

```powershell
& "D:\codex\multimodal-examples\.conda-envs\tld-demo\python.exe" `
  "D:\codex\mllm-demo\tools\build_tiny_dataset.py"
```

## 2. 安装当前项目

```powershell
& "D:\codex\multimodal-examples\.conda-envs\tld-demo\python.exe" -m pip install -e D:\codex\mllm-demo
```

如果你后面想接 Hugging Face 数据集，再额外补：

```powershell
& "D:\codex\multimodal-examples\.conda-envs\tld-demo\python.exe" -m pip install datasets
```

## 3. 跑最小 caption 阶段

```powershell
& "D:\codex\multimodal-examples\.conda-envs\tld-demo\python.exe" -m mllm_demo.train `
  --train-jsonl D:\codex\mllm-demo\data\tiny\caption.jsonl `
  --output-dir D:\codex\mllm-demo\checkpoints\caption `
  --epochs 1 `
  --batch-size 2 `
  --vision-model google/vit-base-patch16-224-in21k `
  --lm-model distilgpt2
```

## 4. 跑 detection / vqa / multitask

```powershell
& "D:\codex\multimodal-examples\.conda-envs\tld-demo\python.exe" -m mllm_demo.train `
  --train-jsonl D:\codex\mllm-demo\data\tiny\detection.jsonl `
  --output-dir D:\codex\mllm-demo\checkpoints\detection `
  --epochs 1 `
  --batch-size 2
```

```powershell
& "D:\codex\multimodal-examples\.conda-envs\tld-demo\python.exe" -m mllm_demo.train `
  --train-jsonl D:\codex\mllm-demo\data\tiny\vqa.jsonl `
  --output-dir D:\codex\mllm-demo\checkpoints\vqa `
  --epochs 1 `
  --batch-size 2
```

```powershell
& "D:\codex\multimodal-examples\.conda-envs\tld-demo\python.exe" -m mllm_demo.train `
  --train-jsonl D:\codex\mllm-demo\data\tiny\multitask.jsonl `
  --output-dir D:\codex\mllm-demo\checkpoints\multitask `
  --epochs 1 `
  --batch-size 2
```

## 5. 推理

```powershell
& "D:\codex\multimodal-examples\.conda-envs\tld-demo\python.exe" -m mllm_demo.infer `
  --checkpoint-dir D:\codex\mllm-demo\checkpoints\caption `
  --image-path D:\codex\mllm-demo\data\tiny\images\lion.jpg `
  --prompt "Describe the image in one sentence."
```

## 数据格式

每条样本一行 JSON，统一结构如下：

```json
{
  "image": "D:/codex/mllm-demo/data/tiny/images/lion.jpg",
  "task": "caption",
  "prompt": "Describe the image in one sentence.",
  "target": "A lion is lying on the grass."
}
```

这套格式的好处是，后面切到自己的数据时，不需要改训练主逻辑，只要继续产出 `image + prompt + target` 即可。

## 这版代码刻意保留的教学特征

- 视觉编码器和语言模型都来自预训练权重
- 多模态连接层只有一个很小的 `VisionProjector`
- 图像特征以“视觉 token”的形式拼接到文本 embedding 前
- `caption / detection / vqa / multitask` 共享同一个训练入口

## 后续怎么扩展

- 把 `distilgpt2` 替换成 `SmolLM` 等更适合指令跟随的小模型
- 把 tiny 数据换成 Flickr8k、COCO 风格 caption 数据、VQAv2 子集
- 把 detection 的文本目标从单框 JSON 扩到多框列表
- 把图像 token 从 1 个 pooled token 扩成多个 patch token
