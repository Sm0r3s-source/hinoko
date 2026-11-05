# Builds Hinoko and installs it like a normal app: Start Menu + Desktop shortcut.
# Run in PowerShell from this folder:  .\install-windows.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
  Write-Host "Rust is not installed. Get it from https://rustup.rs (download rustup-init.exe), then reopen PowerShell and rerun this."
  exit 1
}
cargo build --release -p hinoko
if ($LASTEXITCODE -ne 0) { Write-Host "Build failed. Copy the error above and send it to me."; exit 1 }
$dir = Join-Path $env:LOCALAPPDATA "Programs\Hinoko"
New-Item -ItemType Directory -Force $dir | Out-Null
Copy-Item "target\release\hinoko.exe" $dir -Force
Copy-Item "assets\hinoko.ico" $dir -Force
$shell = New-Object -ComObject WScript.Shell
foreach ($place in @([Environment]::GetFolderPath("Desktop"), (Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"))) {
  $lnk = $shell.CreateShortcut((Join-Path $place "Hinoko.lnk"))
  $lnk.TargetPath = Join-Path $dir "hinoko.exe"
  $lnk.IconLocation = Join-Path $dir "hinoko.ico"
  $lnk.WorkingDirectory = $dir
  $lnk.Save()
}
Write-Host "Done. Open Hinoko from the Desktop or Start Menu."
