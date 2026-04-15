$env:PYTHONPATH = "D:\codex\mllm-demo\src"
& "D:\codex\multimodal-examples\.conda-envs\tld-demo\python.exe" -m mllm_demo.train `
  --train-jsonl "D:\codex\mllm-demo\data\repo\part2_animals_od\train\data.jsonl" `
  --output-dir "D:\codex\mllm-demo\output\part2" `
  --checkpoint-path "D:\codex\mllm-demo\output\part2\training_checkpoint.pt" `
  --init-from "D:\codex\mllm-demo\output\part1" `
  --vision-model "google/vit-large-patch16-224" `
  --lm-model "HuggingFaceTB/SmolLM-360M" `
  --epochs 50 `
  --batch-size 4 `
  --lr 1e-4 `
  --max-length 256 `
  --early-stop-patience 5 `
  --early-stop-min-delta 0.003
