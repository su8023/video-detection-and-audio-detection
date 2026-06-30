param(
  [string]$Output = "D:\local-ai-monitor-package.zip"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Temp = Join-Path $env:TEMP ("local-ai-monitor-package-" + [guid]::NewGuid().ToString("N"))
$Stage = Join-Path $Temp "local-ai-monitor"

$excludeDirs = @(
  ".venv",
  "data",
  "models",
  "__pycache__",
  ".git",
  ".pytest_cache"
)

$excludeFiles = @(
  "*.pyc",
  "*.pyo",
  "*.part",
  "*.log",
  "*.zip"
)

New-Item -ItemType Directory -Path $Stage | Out-Null

Get-ChildItem -LiteralPath $Root -Force | ForEach-Object {
  if ($excludeDirs -contains $_.Name) {
    return
  }
  $dest = Join-Path $Stage $_.Name
  if ($_.PSIsContainer) {
    Copy-Item -LiteralPath $_.FullName -Destination $dest -Recurse -Force
  } else {
    $skip = $false
    foreach ($pattern in $excludeFiles) {
      if ($_.Name -like $pattern) {
        $skip = $true
        break
      }
    }
    if (-not $skip) {
      Copy-Item -LiteralPath $_.FullName -Destination $dest -Force
    }
  }
}

New-Item -ItemType Directory -Path (Join-Path $Stage "models") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $Stage "data") -Force | Out-Null
New-Item -ItemType File -Path (Join-Path $Stage "models\.gitkeep") -Force | Out-Null
New-Item -ItemType File -Path (Join-Path $Stage "data\.gitkeep") -Force | Out-Null

Get-ChildItem -LiteralPath $Stage -Recurse -Force | Where-Object {
  foreach ($pattern in $excludeFiles) {
    if ($_.Name -like $pattern) {
      return $true
    }
  }
  return $false
} | Remove-Item -Force

if (Test-Path -LiteralPath $Output) {
  Remove-Item -LiteralPath $Output -Force
}

Compress-Archive -Path $Stage -DestinationPath $Output -CompressionLevel Optimal
Remove-Item -LiteralPath $Temp -Recurse -Force

Write-Host "Package created: $Output"
