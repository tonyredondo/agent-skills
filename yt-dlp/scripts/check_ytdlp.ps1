param(
    [switch]$VerboseOutput,
    [switch]$SkipLatestCheck
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

function Convert-YtDlpVersion {
    param(
        [string]$Value
    )

    if ($Value -match '(\d+(?:\.\d+){1,3})') {
        return [version]$Matches[1]
    }

    return $null
}

function Get-LatestYtDlpReleaseVersion {
    try {
        $response = Invoke-RestMethod `
            -Uri "https://api.github.com/repos/yt-dlp/yt-dlp/releases/latest" `
            -Headers @{
                "Accept" = "application/vnd.github+json"
                "User-Agent" = "codex-yt-dlp-skill"
            } `
            -TimeoutSec 15

        if ($response.tag_name) {
            return ($response.tag_name -replace '^v', '').Trim()
        }
    } catch {
        return $null
    }

    return $null
}

Write-Host "Command resolution"
Show-CommandInfo yt-dlp | Format-Table -AutoSize
Show-CommandInfo ffmpeg | Format-Table -AutoSize
Show-CommandInfo ffprobe | Format-Table -AutoSize

Write-Host ""
Write-Host "Runtime versions"
$installedYtDlpVersion = $null
if (Get-Command yt-dlp -ErrorAction SilentlyContinue) {
    $installedYtDlpVersion = (& yt-dlp --version).Trim()
    Write-Host ("yt-dlp: " + $installedYtDlpVersion)
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

if ($SkipLatestCheck) {
    Write-Host ""
    Write-Host "yt-dlp release check"
    Write-Host "Skipped because -SkipLatestCheck was passed."
} else {
    Write-Host ""
    Write-Host "yt-dlp release check"
    if (-not $installedYtDlpVersion) {
        Write-Host "Installed version: not found"
    } else {
        Write-Host "Installed version: $installedYtDlpVersion"
    }

    $latestYtDlpVersion = Get-LatestYtDlpReleaseVersion
    if (-not $latestYtDlpVersion) {
        Write-Host "Latest GitHub release: unavailable; network or GitHub API access may be blocked."
    } else {
        Write-Host "Latest GitHub release: $latestYtDlpVersion"

        $installedComparable = Convert-YtDlpVersion $installedYtDlpVersion
        $latestComparable = Convert-YtDlpVersion $latestYtDlpVersion

        if (-not $installedYtDlpVersion) {
            Write-Host "Version status: install yt-dlp before running download actions."
        } elseif ($installedComparable -and $latestComparable) {
            if ($installedComparable -lt $latestComparable) {
                Write-Host "Version status: update recommended before using yt-dlp."
                Write-Host "Update through the same route used for installation."
            } elseif ($installedComparable -gt $latestComparable) {
                Write-Host "Version status: installed version is newer than the latest stable release."
            } else {
                Write-Host "Version status: installed version matches the latest GitHub release."
            }
        } elseif ($installedYtDlpVersion -ne $latestYtDlpVersion) {
            Write-Host "Version status: installed version differs from the latest GitHub release."
        } else {
            Write-Host "Version status: installed version matches the latest GitHub release."
        }
    }
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
