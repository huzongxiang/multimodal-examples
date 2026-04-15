$env:PYTHONPATH = "D:\codex\mllm-demo\src"
& "D:\codex\multimodal-examples\.conda-envs\tld-demo\python.exe" -m mllm_demo.train `
  --train-jsonl "D:\codex\mllm-demo\data\repo\part1_flickr8k\train.jsonl" `
  --output-dir "D:\codex\mllm-demo\output\part1" `
  --checkpoint-path "D:\codex\mllm-demo\output\part1\training_checkpoint.pt" `
  --vision-model "google/vit-large-patch16-224" `
  --lm-model "HuggingFaceTB/SmolLM-360M" `
  --epochs 50 `
  --batch-size 4 `
  --lr 2e-4 `
  --max-length 64 `
  --early-stop-patience 8 `
  --early-stop-min-delta 0.005
