$env:PYTHONPATH = "D:\codex\mllm-demo\src"
& "D:\codex\multimodal-examples\.conda-envs\tld-demo\python.exe" -m mllm_demo.train `
  --train-jsonl "D:\codex\mllm-demo\data\tiny\multitask.jsonl" `
  --output-dir "D:\codex\mllm-demo\checkpoints\multitask" `
  --epochs 1 `
  --batch-size 2 `
  --vision-model "google/vit-base-patch16-224-in21k" `
  --lm-model "distilgpt2"
