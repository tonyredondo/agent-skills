param(
    [switch]$VerboseOutput
)

$ErrorActionPreference = "Stop"

function Show-CommandInfo {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    $commands = Get-Command $Name -All -ErrorAction SilentlyContinue
    if (-not $commands) {
        [pscustomobject]@{
            Command = $Name
            Found = $false
            Source = ""
            Version = ""
        }
        return
    }

    foreach ($command in $commands) {
        [pscustomobject]@{
            Command = $Name
            Found = $true
            Source = $command.Source
            Version = $command.Version
        }
    }
}

function Find-UserPathExecutable {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    [Environment]::GetEnvironmentVariable("Path", "User") -split ";" |
        Where-Object { $_ } |
        ForEach-Object {
            $candidate = Join-Path $_ $Name
            if (Test-Path -LiteralPath $candidate) {
                $candidate
            }
        }
}

Write-Host "Command resolution"
Show-CommandInfo yt-dlp | Format-Table -AutoSize
Show-CommandInfo ffmpeg | Format-Table -AutoSize
Show-CommandInfo ffprobe | Format-Table -AutoSize

Write-Host ""
Write-Host "Runtime versions"
if (Get-Command yt-dlp -ErrorAction SilentlyContinue) {
    Write-Host ("yt-dlp: " + (& yt-dlp --version))
} else {
    Write-Host "yt-dlp: not found"
}

if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
    Write-Host ("ffmpeg: " + ((& ffmpeg -version | Select-Object -First 1) -replace "`r", ""))
} else {
    Write-Host "ffmpeg: not found"
}

if (Get-Command ffprobe -ErrorAction SilentlyContinue) {
    Write-Host ("ffprobe: " + ((& ffprobe -version | Select-Object -First 1) -replace "`r", ""))
} else {
    Write-Host "ffprobe: not found"
}

$userFfmpeg = @(Find-UserPathExecutable "ffmpeg.exe")
$userFfprobe = @(Find-UserPathExecutable "ffprobe.exe")
if ($userFfmpeg.Count -gt 0 -or $userFfprobe.Count -gt 0) {
    Write-Host ""
    Write-Host "Executables visible in the User PATH registry value"
    foreach ($path in $userFfmpeg) {
        Write-Host "ffmpeg candidate: $path"
    }
    foreach ($path in $userFfprobe) {
        Write-Host "ffprobe candidate: $path"
    }
}

if ((-not (Get-Command ffprobe -ErrorAction SilentlyContinue)) -and $userFfprobe.Count -gt 0) {
    Write-Host ""
    Write-Host "Note: ffprobe exists in the User PATH registry value but is not visible in this shell. Restart PowerShell."
}

$currentFfmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
if ($currentFfmpeg -and $userFfmpeg.Count -gt 0 -and $userFfmpeg -notcontains $currentFfmpeg.Source) {
    Write-Host ""
    Write-Host "Note: this shell resolves ffmpeg to a different executable than the ffmpeg found in User PATH."
    Write-Host "Use --ffmpeg-location or adjust PATH order if yt-dlp should use the User PATH ffmpeg."
}

if ($VerboseOutput) {
    Write-Host ""
    Write-Host "User PATH entries mentioning yt-dlp, FFmpeg, or WinGet"
    [Environment]::GetEnvironmentVariable("Path", "User") -split ";" |
        Where-Object { $_ -match "yt-dlp|ffmpeg|WinGet" } |
        ForEach-Object { Write-Host $_ }
}
