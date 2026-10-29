# Omnicompute one-line installer for Windows. Works from any network:
#   $env:OMNICOMPUTE_CODE="CODE"; irm https://raw.githubusercontent.com/TOS-main/Omnicompute/main/omnicompute/installers/install.ps1 | iex
# (CODE = the contribute code or permanent link the main device shows), or served by the main device:
#   irm "@ORIGIN@/install.ps1?c=@KEY@" | iex
# Without a code it asks: join my mesh, start my own mesh here ($env:OMNICOMPUTE_HOST="1"), or help
# the creator's public mesh ($env:OMNICOMPUTE_CODE="creator").
# $env:OMNICOMPUTE_CORES=N picks how many CPU cores to use without asking; $env:OMNICOMPUTE_YES="1" answers yes.
# Installs Python if needed (with winget), installs Omnicompute into its own folder, and starts
# contributing in the background with no window. It starts again at every login.
# The main computer can remove it again (Devices tab > Remove), which uninstalls it here.
# It asks before installing. Remove it any time with:
#   & "$env:LOCALAPPDATA\Omnicompute\venv\Scripts\omnicompute.exe" contribute --uninstall

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$Hub = "@ORIGIN@"
$Key = "@KEY@"
$Dir = Join-Path $env:LOCALAPPDATA "Omnicompute"
$Repo = if ($env:OMNICOMPUTE_REPO) { $env:OMNICOMPUTE_REPO } else { "TOS-main/Omnicompute" }
if ($Hub.StartsWith("@")) {  # run straight from GitHub: the code finds the main device wherever it is
  $Hub = ""
  $Code = $env:OMNICOMPUTE_CODE
  $HostMode = $env:OMNICOMPUTE_HOST -eq "1"
  if (-not $Code -and -not $HostMode) {
    Write-Host "What should this computer do?"
    Write-Host "  1) Help my own mesh (I have a code or link from my main device)"
    Write-Host "  2) Start my own mesh here (this computer becomes the main device)"
    Write-Host "  3) Help the Omnicompute creator's public mesh with spare power"
    $choice = Read-Host "Choose 1, 2 or 3"
    if ($choice -eq "1") { $Code = Read-Host "Paste the code or link" }
    elseif ($choice -eq "2") { $HostMode = $true }
    elseif ($choice -eq "3") { $Code = "creator" }
    if (-not $Code -and -not $HostMode) { Write-Host "Nothing was installed."; return }
  }
  $From = "GitHub ($Repo)"
  if ($HostMode) { $Join = ""; $Where = "a new mesh of your own" }
  elseif ($Code -eq "creator") { $Join = "creator"; $Where = "the Omnicompute creator's public mesh" }
  else { $Join = $Code.Trim(); $Where = "your mesh" }
} else {
  $Join = "$Hub/contribute?c=$Key"; $From = "your main device"; $Where = $Hub
}
$CoresAll = [Environment]::ProcessorCount
New-Item -ItemType Directory -Force -Path $Dir | Out-Null
# lets "Remove device" on the main computer delete this folder again
Set-Content -Path (Join-Path $Dir "installed-by-edgemesh-installer") -Value "Omnicompute installer"

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

if (-not $Join) {
  Write-Host "Omnicompute will make this computer the main device of $Where."
  Write-Host "  - installs into $Dir (no admin rights needed)"
  Write-Host "  - then starts it in this window; close the window to stop, and start it again from the"
  Write-Host "    'Omnicompute host' shortcut on your desktop"
} else {
Write-Host "Omnicompute will set up this computer to lend its spare power to $Where."
Write-Host "  - installs into $Dir (no admin rights needed)"
Write-Host "  - runs in the background with no window (it shows in Task Manager as pythonw / Omnicompute)"
Write-Host "  - starts when you sign in, uses at most 80% by default, and pauses when the main computer says so"
Write-Host "  - stop and remove it any time:  & `"$Dir\venv\Scripts\omnicompute.exe`" contribute --uninstall"
}
Write-Host "Only do this on a computer you own or have permission to use."
if ($env:OMNICOMPUTE_YES -ne "1") {
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

$Cores = $env:OMNICOMPUTE_CORES
if (-not $Cores -and $env:OMNICOMPUTE_YES -ne "1") { $Cores = Read-Host "This computer has $CoresAll CPU cores. How many should Omnicompute use? [all]" }
$Awake = $env:OMNICOMPUTE_AWAKE
if (-not $Awake -and $env:OMNICOMPUTE_YES -ne "1") { $Awake = Read-Host "Keep this computer from going to sleep while it helps (the screen can still turn off)? [y/N]" }
$awakeArgs = if ("$Awake" -match "^[yY1]") { @("--awake") } else { @() }
$n = 0
if (-not ([int]::TryParse("$Cores", [ref]$n)) -or $n -lt 1 -or $n -ge $CoresAll) { $n = 0 }

Write-Host "Downloading Omnicompute from $From..."
$zip = Join-Path $Dir "omnicompute.zip"
$url = if ($Hub) { "$Hub/download/omnicompute.zip?c=$Key" } else { "https://github.com/$Repo/archive/refs/heads/main.zip" }
Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $zip
$src = Join-Path $Dir "src"
if (Test-Path $src) { Remove-Item -Recurse -Force $src }
Expand-Archive -Path $zip -DestinationPath $src -Force
$pkg = (Get-ChildItem -Path $src -Directory | Where-Object { Test-Path (Join-Path $_.FullName "pyproject.toml") } | Select-Object -First 1).FullName

$venv = Join-Path $Dir "venv"
if (-not (Test-Path (Join-Path $venv "Scripts\python.exe"))) {
  $exe = $py[0]; $rest = @($py | Select-Object -Skip 1)
  & $exe @rest -m venv $venv
}
$vpy = Join-Path $venv "Scripts\python.exe"
Write-Host "Installing (this takes a minute the first time)..."
& $vpy -m pip install --quiet --disable-pip-version-check --upgrade $pkg
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

# NVIDIA card: offer CUDA (PyTorch), which runs the model several times faster than DirectX 12
$nvidia = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | Where-Object { $_.Name -match "NVIDIA" } | Select-Object -First 1
if ($nvidia) {
  $ErrorActionPreference = "Continue"  # pip and python write progress to stderr; that's not a failure
  $hasCuda = $false
  try { & $vpy -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>$null; $hasCuda = ($LASTEXITCODE -eq 0) } catch {}
  if (-not $hasCuda) {
    $cuda = if ($env:OMNICOMPUTE_YES -eq "1") { "y" } else { Read-Host "Found $($nvidia.Name). Install CUDA support for it (about 3 GB download, much faster)? [Y/n]" }
    if ($cuda -notmatch "^[nN]") {
      Write-Host "Installing CUDA support (PyTorch); this can take a few minutes..."
      & $vpy -m pip install --quiet --disable-pip-version-check torch --index-url https://download.pytorch.org/whl/cu126
      if ($LASTEXITCODE -ne 0) { Write-Host "CUDA support could not be installed; Omnicompute will use DirectX 12 for the GPU instead." }
    }
  }
  $ErrorActionPreference = "Stop"
}

$coreArgs = if ($n -gt 0) { @("--cores", "$n") } else { @() }
if (-not $Join) {
  if ($env:OMNICOMPUTE_KEY) { $coreArgs = @("--key", $env:OMNICOMPUTE_KEY) + $coreArgs }  # keep an existing mesh
  $exe = Join-Path $venv "Scripts\omnicompute.exe"
  $lnk = Join-Path ([Environment]::GetFolderPath("Desktop")) "Omnicompute host.lnk"
  try {
    $s = (New-Object -ComObject WScript.Shell).CreateShortcut($lnk)
    $s.TargetPath = $exe; $s.Arguments = (@("host", "--public") + $coreArgs) -join " "; $s.Save()
  } catch {}
  Write-Host ""
  Write-Host "Starting your mesh. Share the permanent contribute link it prints; it keeps working when"
  Write-Host "this computer's address changes. Close this window to stop."
  & $exe host --public @coreArgs
  return
}
& $vpy -m omnicompute contribute $Join @coreArgs @awakeArgs
Write-Host ""
Write-Host "Done. This computer now shows up in the main device's Devices tab, from any network."
Write-Host "Usage limit is 80% by default; change it and the cores there. On this computer:"
Write-Host "  change:     & `"$venv\Scripts\omnicompute.exe`" limits --cpu 50 --gpu 80 --cores 4 --awake on"
Write-Host "  pause:      & `"$venv\Scripts\omnicompute.exe`" contribute --stop"
Write-Host "  uninstall:  & `"$venv\Scripts\omnicompute.exe`" contribute --uninstall"
