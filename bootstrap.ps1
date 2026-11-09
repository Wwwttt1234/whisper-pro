$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$taskProjectRoot = $PSScriptRoot
$taskData = if ($env:WHISPER_PRO_DATA_DIR) { $env:WHISPER_PRO_DATA_DIR } else { Join-Path $env:LOCALAPPDATA 'WhisperPro' }
$taskEnv = Join-Path $taskData 'app-environment-v2'
$taskStamp = Join-Path $taskEnv '.whisper-setup.json'
$taskLock = Join-Path $taskProjectRoot 'uv.lock'
foreach ($taskFile in @('uv.lock', 'pyproject.toml', 'launcher.py', 'app.py')) {
    if (!(Test-Path -LiteralPath (Join-Path $taskProjectRoot $taskFile) -PathType Leaf)) {
        throw 'Extract the entire Whisper Pro archive before opening Start.bat.'
    }
}
if ($env:PROCESSOR_ARCHITECTURE -ne 'AMD64' -or $env:PROCESSOR_ARCHITEW6432 -eq 'ARM64') {
    throw 'This Windows package requires Windows x64.'
}
foreach ($taskDll in @('msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll')) {
    if (!(Test-Path -LiteralPath (Join-Path $env:windir "System32\$taskDll"))) {
        throw 'Install Microsoft Visual C++ Redistributable (x64), then open Start.bat again. https://aka.ms/vc14/vc_redist.x64.exe'
    }
}
# Use .NET hashing so setup also works in Windows PowerShell without Get-FileHash.
$taskHasher = [System.Security.Cryptography.SHA256]::Create()
try { $taskDigest = [System.BitConverter]::ToString($taskHasher.ComputeHash([System.IO.File]::ReadAllBytes($taskLock))).Replace('-', '').ToLowerInvariant() }
finally { $taskHasher.Dispose() }
$taskReady = (Test-Path -LiteralPath (Join-Path $taskEnv 'Scripts\python.exe')) -and (Test-Path -LiteralPath $taskStamp) -and ([System.IO.File]::ReadAllText($taskStamp).Trim() -eq ('"' + $taskDigest + '"'))
if (!$taskReady) {
    Write-Host "Setup will use Python 3.12 and install the application libraries in $taskData."
    Write-Host 'Keep your Internet connection on during setup. The app opens when setup finishes.'
    if ($env:WHISPER_PRO_SETUP_YES -ne '1') {
        Write-Host 'Set up and launch? [y/N]'
        if ([Console]::ReadLine() -notmatch '^y(es)?$') { Write-Host 'Setup cancelled.'; exit 0 }
    }
}
$taskToolRoot = Join-Path $taskData 'tools\uv-0.8.14'
New-Item -ItemType Directory -Force -Path $taskToolRoot | Out-Null
$taskUv = Join-Path $taskToolRoot 'uv.exe'
if (!(Test-Path -LiteralPath $taskUv)) {
    Write-Host 'Preparing Whisper Pro. The first setup needs an Internet connection.'
    $taskArch = if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') { 'aarch64' } else { 'x86_64' }
    $taskAssetName = "uv-$taskArch-pc-windows-msvc.zip"
    $taskRelease = Invoke-RestMethod -Uri 'https://api.github.com/repos/astral-sh/uv/releases/tags/0.8.14' -Headers @{ 'User-Agent' = 'WhisperPro' } -TimeoutSec 60
    $taskAsset = $taskRelease.assets | Where-Object name -eq $taskAssetName | Select-Object -First 1
    if (!$taskAsset -or !$taskAsset.digest) { throw 'No verified setup tool is available for this computer.' }
    $taskArchive = Join-Path $taskToolRoot $taskAssetName
    if (!(Test-Path -LiteralPath $taskArchive)) {
        Invoke-WebRequest -UseBasicParsing -Uri $taskAsset.browser_download_url -OutFile $taskArchive -TimeoutSec 600
    }
    $taskHasher = [System.Security.Cryptography.SHA256]::Create()
    $taskStream = [System.IO.File]::OpenRead($taskArchive)
    try { $taskHash = [System.BitConverter]::ToString($taskHasher.ComputeHash($taskStream)).Replace('-', '').ToLowerInvariant() }
    finally { $taskStream.Dispose(); $taskHasher.Dispose() }
    if ($taskHash -ne $taskAsset.digest.Replace('sha256:', '')) {
        Remove-Item -LiteralPath $taskArchive -Force
        throw 'The setup download failed verification. Please retry.'
    }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $taskZip = [System.IO.Compression.ZipFile]::OpenRead($taskArchive)
    $taskSafeRoot = [System.IO.Path]::GetFullPath($taskToolRoot).TrimEnd('\') + '\'
    try {
        foreach ($taskEntry in $taskZip.Entries) {
            $taskEntryPath = [System.IO.Path]::GetFullPath((Join-Path $taskToolRoot $taskEntry.FullName))
            if (!$taskEntryPath.StartsWith($taskSafeRoot, [System.StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe archive path.' }
            if (!$taskEntry.Name) { [System.IO.Directory]::CreateDirectory($taskEntryPath) | Out-Null }
            else {
                [System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($taskEntryPath)) | Out-Null
                [System.IO.Compression.ZipFileExtensions]::ExtractToFile($taskEntry, $taskEntryPath, $true)
            }
        }
    } finally { $taskZip.Dispose() }
    if (!(Test-Path -LiteralPath $taskUv)) {
        $taskFound = Get-ChildItem -LiteralPath $taskToolRoot -Filter uv.exe -Recurse -File | Select-Object -First 1
        if (!$taskFound) { throw 'The setup tool was not found in its verified package.' }
        Copy-Item -LiteralPath $taskFound.FullName -Destination $taskUv
    }
}
$env:WHISPER_PRO_DATA_DIR = $taskData
$env:UV_PROJECT_ENVIRONMENT = $taskEnv
if (!$env:UV_CACHE_DIR) { $env:UV_CACHE_DIR = Join-Path $taskData 'cache\uv' }
if (!$env:UV_HTTP_TIMEOUT) { $env:UV_HTTP_TIMEOUT = '120' }
$env:PYTHONUTF8 = '1'
Write-Host 'Opening Whisper Pro. Existing installations are reused.'
& $taskUv run --project $taskProjectRoot --locked --no-dev --python 3.12 python (Join-Path $taskProjectRoot 'launcher.py')
exit $LASTEXITCODE
