# 模型代码总览

## 文档定位

本文档用于汇总当前本地教学复现版 `vlm-from-scratch` 的核心代码结构，作为后续章节写作的基础材料，而不是逐行源码注释。

如果需要查看完整核心源码展开版，请直接参考：

- `docs\model_full_code.md`

## 涉及文件

- `src\mllm_demo\model.py`
- `src\mllm_demo\data.py`
- `src\mllm_demo\train.py`
- `src\mllm_demo\eval_utils.py`
- `src\mllm_demo\eval.py`
- `src\mllm_demo\infer.py`

## 代码层面的核心结论

当前项目在“实验阶段”上分成了 `Part 1 / Part 2 / Part 3 / Part 4`，但在“模型实现”层面，真正的核心模型代码只有一套：

- `MiniVLM`
- `VisionProjector`
- 统一 JSONL 数据管线
- 统一训练循环
- 统一推理与评估逻辑

各阶段之间的主要差别不是模型拓扑完全不同，而是：

- 视觉编码器规格不同；
- 语言模型规格不同；
- 数据集不同；
- prompt 与目标文本格式不同；
- 训练脚本参数不同。

因此，后续写作时最稳的表述是：

**本实验不是为每个阶段分别设计一套独立模型，而是在统一最小多模态骨架上，通过任务文本化和阶段化训练实现能力扩展。**

## 推荐使用方式

- 如果要快速理解结构，请先看本文件；
- 如果要直接引用完整源码，请看 `docs\model_full_code.md`；
- 如果要写章节正文，可以把本文件当成结构纲要，把 `model_full_code.md` 当成附录型原始材料。
