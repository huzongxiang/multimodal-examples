param(
  [string]$MetricsPath,
  [string]$OutputPath
)

& "D:\codex\multimodal-examples\.conda-envs\tld-demo\python.exe" `
  "D:\codex\mllm-demo\tools\plot_metrics.py" `
  --metrics-path $MetricsPath `
  --output-path $OutputPath
