$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:HF_XET_HIGH_PERFORMANCE = "1"
$env:HF_HUB_DOWNLOAD_TIMEOUT = "120"

$RequiredFiles = @(
    @{ Path = "$ProjectRoot\ckpts\wan_animate_2\wan_animate_2_bf16_distillation.safetensors"; Size = 32789901704 },
    @{ Path = "$ProjectRoot\ckpts\videomodel\Wan-AI\models_t5_umt5-xxl-enc-bf16.pth"; Size = 11361920418 },
    @{ Path = "$ProjectRoot\ckpts\videomodel\Wan-AI\models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"; Size = 4772359047 },
    @{ Path = "$ProjectRoot\ckpts\videomodel\Wan-AI\vae.pth"; Size = 783806166 }
)

$Missing = @($RequiredFiles | Where-Object {
    -not (Test-Path -LiteralPath $_.Path) -or (Get-Item -LiteralPath $_.Path).Length -ne $_.Size
})

if ($Missing.Count -eq 0) {
    Write-Host "Wan-Animate-2 distilled model files are already complete."
    exit 0
}

& "$ProjectRoot\.venv\Scripts\hf.exe" download `
    Wan-AI/Wan2.2-Animate-2-14B `
    --local-dir "$ProjectRoot\ckpts" `
    --exclude "wan_animate_2/wan_animate_2_bf16.safetensors" `
    --max-workers 4
