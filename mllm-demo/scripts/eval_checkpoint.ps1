param(
  [string]$CheckpointDir,
  [string]$EvalJsonl,
  [string]$OutputDir = "",
  [int]$MaxNewTokens = 64
)

$env:PYTHONPATH = "D:\codex\mllm-demo\src"
& "D:\codex\multimodal-examples\.conda-envs\tld-demo\python.exe" -m mllm_demo.eval `
  --checkpoint-dir $CheckpointDir `
  --eval-jsonl $EvalJsonl `
  --output-dir $OutputDir `
  --max-new-tokens $MaxNewTokens
