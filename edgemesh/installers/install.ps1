# EdgeMesh one-line installer for Windows, served by your main computer.
#   irm "@ORIGIN@/install.ps1?c=@KEY@" | iex
# Installs Python if needed (with winget), installs EdgeMesh into its own folder, and starts
# contributing in the background with no window. It starts again at every login.
# The main computer can remove it again (Devices tab > Remove), which uninstalls it here.
# It asks before installing. Remove it any time with:
#   & "$env:LOCALAPPDATA\EdgeMesh\venv\Scripts\edgemesh.exe" contribute --uninstall

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$Hub = "@ORIGIN@"
$Key = "@KEY@"
$Dir = Join-Path $env:LOCALAPPDATA "EdgeMesh"
New-Item -ItemType Directory -Force -Path $Dir | Out-Null
# lets "Remove device" on the main computer delete this folder again
Set-Content -Path (Join-Path $Dir "installed-by-edgemesh-installer") -Value "EdgeMesh installer"

function Find-Python {
  foreach ($c in @("py", "python", "python3")) {
    $cmd = Get-Command $c -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source -notlike "*WindowsApps*") {
      $extra = if ($c -eq "py") { @("-3") } else { @() }
      try {
        & $cmd.Source @extra -c "import sys; assert sys.version_info >= (3, 10)" 2>$null
        if ($LASTEXITCODE -eq 0) { $found = @($cmd.Source) + $extra; return ,$found }
      } catch {}
    }
  }
  return $null
}

Write-Host "EdgeMesh will set up this computer to lend its spare power to the AI mesh at $Hub."
Write-Host "  - installs into $Dir (no admin rights needed)"
Write-Host "  - runs in the background with no window (it shows in Task Manager as pythonw / EdgeMesh)"
Write-Host "  - starts when you sign in, uses at most 80% by default, and pauses when the main computer says so"
Write-Host "  - stop and remove it any time:  & `"$Dir\venv\Scripts\edgemesh.exe`" contribute --uninstall"
Write-Host "Only do this on a computer you own or have permission to use."
if ($env:EDGEMESH_YES -ne "1") {
  $answer = Read-Host "Continue? [y/N]"
  if ($answer -notmatch "^[yY]") { Write-Host "Nothing was installed."; return }
}
$py = Find-Python
if (-not $py) {
  Write-Host "Installing Python 3.12 (one time)..."
  winget install -e --id Python.Python.3.12 --scope user --silent --accept-package-agreements --accept-source-agreements | Out-Null
  $env:Path = [Environment]::GetEnvironmentVariable("Path", "User") + ";" + [Environment]::GetEnvironmentVariable("Path", "Machine")
  $py = Find-Python
  if (-not $py) { throw "Python could not be installed automatically. Install it from python.org, then run this command again." }
}

Write-Host "Downloading EdgeMesh from your main computer..."
$zip = Join-Path $Dir "edgemesh.zip"
Invoke-WebRequest -UseBasicParsing -Uri "$Hub/download/edgemesh.zip?c=$Key" -OutFile $zip
$src = Join-Path $Dir "src"
if (Test-Path $src) { Remove-Item -Recurse -Force $src }
Expand-Archive -Path $zip -DestinationPath $src -Force

$venv = Join-Path $Dir "venv"
if (-not (Test-Path (Join-Path $venv "Scripts\python.exe"))) {
  $exe = $py[0]; $rest = @($py | Select-Object -Skip 1)
  & $exe @rest -m venv $venv
}
$vpy = Join-Path $venv "Scripts\python.exe"
Write-Host "Installing (this takes a minute the first time)..."
& $vpy -m pip install --quiet --disable-pip-version-check --upgrade (Join-Path $src "edgemesh-src")
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

# NVIDIA card: offer CUDA (PyTorch), which runs the model several times faster than DirectX 12
$nvidia = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | Where-Object { $_.Name -match "NVIDIA" } | Select-Object -First 1
if ($nvidia) {
  $ErrorActionPreference = "Continue"  # pip and python write progress to stderr; that's not a failure
  $hasCuda = $false
  try { & $vpy -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>$null; $hasCuda = ($LASTEXITCODE -eq 0) } catch {}
  if (-not $hasCuda) {
    $cuda = if ($env:EDGEMESH_YES -eq "1") { "y" } else { Read-Host "Found $($nvidia.Name). Install CUDA support for it (about 3 GB download, much faster)? [Y/n]" }
    if ($cuda -notmatch "^[nN]") {
      Write-Host "Installing CUDA support (PyTorch); this can take a few minutes..."
      & $vpy -m pip install --quiet --disable-pip-version-check torch --index-url https://download.pytorch.org/whl/cu126
      if ($LASTEXITCODE -ne 0) { Write-Host "CUDA support could not be installed; EdgeMesh will use DirectX 12 for the GPU instead." }
    }
  }
  $ErrorActionPreference = "Stop"
}

& $vpy -m edgemesh contribute "$Hub/contribute?c=$Key"
Write-Host ""
Write-Host "Done. This computer now shows up in the main computer's Devices tab."
Write-Host "Usage limit is 80% by default; change it there. On this computer:"
Write-Host "  pause:      & `"$venv\Scripts\edgemesh.exe`" contribute --stop"
Write-Host "  uninstall:  & `"$venv\Scripts\edgemesh.exe`" contribute --uninstall"
