param(
  [string]$ImagePath = "D:\codex\mllm-demo\data\tiny\images\lion.jpg",
  [string]$Prompt = "Describe the image in one sentence."
)

$env:PYTHONPATH = "D:\codex\mllm-demo\src"
& "D:\codex\multimodal-examples\.conda-envs\tld-demo\python.exe" -m mllm_demo.infer `
  --checkpoint-dir "D:\codex\mllm-demo\output\part1" `
  --image-path $ImagePath `
  --prompt $Prompt
