# Run U-Net Full-Scene Inference
param (
    [string]$Image = "sample.tif",
    [string]$Label = "",
    [string]$Checkpoint = "checkpoints/best_val_iou.pth",
    [string]$OutputDir = "outputs/unet_inference"
)

Write-Host "Running FloodPlanet U-Net Inference..." -ForegroundColor Cyan
Write-Host "Image: $Image"
Write-Host "Checkpoint: $Checkpoint"
Write-Host "Output Directory: $OutputDir"

$cmdArgs = @("scripts/predict_floodplanet.py", "--image", $Image, "--checkpoint", $Checkpoint, "--output-dir", $OutputDir)
if ($Label -ne "") {
    $cmdArgs += @("--label", $Label)
}

python @cmdArgs
