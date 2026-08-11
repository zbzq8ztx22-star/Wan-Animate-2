$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = $ProjectRoot
$env:PYTHONUTF8 = "1"

Set-Location "$ProjectRoot\infer"
& "$ProjectRoot\.venv\Scripts\python.exe" `
    "$ProjectRoot\infer\wan_animate_2_gradio_distillation.py" `
    --config "$ProjectRoot\infer\wan_animate_2_distillation.yaml" `
    --output-dir "$ProjectRoot\outputs" `
    @args
