$env:PYTHONPATH = "D:\codex\mllm-demo\src"
& "D:\codex\multimodal-examples\.conda-envs\tld-demo\python.exe" -m mllm_demo.train `
  --train-jsonl "D:\codex\mllm-demo\data\repo\part4_multitask\train.jsonl" `
  --output-dir "D:\codex\mllm-demo\output\part4" `
  --checkpoint-path "D:\codex\mllm-demo\output\part4\training_checkpoint.pt" `
  --init-from "D:\codex\mllm-demo\output\part1_for_part4" `
  --vision-model "google/vit-base-patch16-224" `
  --lm-model "HuggingFaceTB/SmolLM-135M" `
  --epochs 50 `
  --batch-size 4 `
  --lr 1e-4 `
  --max-length 256 `
  --early-stop-patience 5 `
  --early-stop-min-delta 0.005
