[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Tools installed through winget. Keep command names separate from package ids because
# some winget packages expose aliases that differ from their package name.
$WingetTools = @(
    @{ Id = "Git.Git"; Command = "git"; VersionArgs = @("--version") },
    @{ Id = "GitHub.cli"; Command = "gh"; VersionArgs = @("--version") },
    @{ Id = "BurntSushi.ripgrep.MSVC"; Command = "rg"; VersionArgs = @("--version") },
    @{ Id = "Microsoft.PowerShell"; Command = "pwsh"; VersionArgs = @("--version") },
    @{ Id = "OpenJS.NodeJS.LTS"; Command = "node"; VersionArgs = @("--version") },
    @{ Id = "OpenJS.NodeJS.LTS"; Command = "npm"; VersionArgs = @("--version") },
    @{ Id = "jqlang.jq"; Command = "jq"; VersionArgs = @("--version") },
    @{ Id = "MikeFarah.yq"; Command = "yq"; VersionArgs = @("--version") },
    @{ Id = "sharkdp.fd"; Command = "fd"; VersionArgs = @("--version") },
    @{ Id = "junegunn.fzf"; Command = "fzf"; VersionArgs = @("--version") },
    @{ Id = "sharkdp.bat"; Command = "bat"; VersionArgs = @("--version") },
    @{ Id = "dandavison.delta"; Command = "delta"; VersionArgs = @("--version") },
    @{ Id = "7zip.7zip"; Command = "7z"; VersionArgs = @() },
    @{ Id = "Casey.Just"; Command = "just"; VersionArgs = @("--version") },
    @{ Id = "koalaman.shellcheck"; Command = "shellcheck"; VersionArgs = @("--version") },
    @{ Id = "mvdan.shfmt"; Command = "shfmt"; VersionArgs = @("--version") },
    @{ Id = "hadolint.hadolint"; Command = "hadolint"; VersionArgs = @("--version") },
    @{ Id = "Kitware.CMake"; Command = "cmake"; VersionArgs = @("--version") },
    @{ Id = "Ninja-build.Ninja"; Command = "ninja"; VersionArgs = @("--version") },
    @{ Id = "pnpm.pnpm"; Command = "pnpm"; VersionArgs = @("--version") },
    @{ Id = "Gitleaks.Gitleaks"; Command = "gitleaks"; VersionArgs = @("version") }
)

# Tools installed through pipx so each Python CLI has isolated dependencies.
$PipxTools = @(
    @{ Package = "ruff"; Command = "ruff"; VersionArgs = @("--version") },
    @{ Package = "pytest"; Command = "pytest"; VersionArgs = @("--version") },
    @{ Package = "mypy"; Command = "mypy"; VersionArgs = @("--version") },
    @{ Package = "pre-commit"; Command = "pre-commit"; VersionArgs = @("--version") },
    @{ Package = "pip-audit"; Command = "pip-audit"; VersionArgs = @("--version") }
)

function Add-UserPathEntry {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [switch]$Create
    )

    if ($Create -and -not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Force -Path $Path | Out-Null
    }

    if (-not (Test-Path -LiteralPath $Path)) {
        Write-Host "PATH entry target does not exist yet, skipping: $Path"
        return
    }

    $currentUserPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $parts = @()
    if (-not [string]::IsNullOrWhiteSpace($currentUserPath)) {
        $parts = $currentUserPath -split ";" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    }

    foreach ($part in $parts) {
        $trimChars = [char[]]@("\", "/")
        if ([string]::Equals($part.TrimEnd($trimChars), $Path.TrimEnd($trimChars), [StringComparison]::OrdinalIgnoreCase)) {
            return
        }
    }

    $parts += $Path
    [Environment]::SetEnvironmentVariable("Path", ($parts -join ";"), "User")
}

function Update-SessionPath {
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $env:Path = "$userPath;$machinePath"
}

function Test-IsAdministrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Invoke-CommandChecked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,

        [string[]]$Arguments = @()
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($Arguments -join ' ')"
    }
}

function Test-CommandHealth {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,

        [string[]]$VersionArgs = @()
    )

    if (-not (Get-Command $Command -ErrorAction SilentlyContinue)) {
        return $false
    }

    if ($VersionArgs.Count -gt 0) {
        $commandLine = @($Command) + $VersionArgs
        & cmd.exe /d /c ($commandLine -join " ") *> $null
        return $LASTEXITCODE -eq 0
    }

    if ($Command -eq "7z") {
        & cmd.exe /d /c $Command *> $null
        return $LASTEXITCODE -eq 0
    }

    return $true
}

function Get-PythonCommand {
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        return $python.Source
    }

    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        return $py.Source
    }

    $knownPythonPaths = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python313\python.exe"),
        "C:\Program Files\Python313\python.exe"
    )

    foreach ($knownPythonPath in $knownPythonPaths) {
        if (Test-Path -LiteralPath $knownPythonPath) {
            return $knownPythonPath
        }
    }

    throw "Python was not found on PATH. Install Python first, then rerun this skill."
}

function Install-WingetIfMissing {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        return
    }

    $windowsAppsPath = Join-Path $env:LOCALAPPDATA "Microsoft\WindowsApps"
    Add-UserPathEntry -Path $windowsAppsPath
    Update-SessionPath

    if (Get-Command winget -ErrorAction SilentlyContinue) {
        return
    }

    Write-Host "winget was not found. Requesting App Installer registration."
    try {
        Add-AppxPackage -RegisterByFamilyName -MainPackage "Microsoft.DesktopAppInstaller_8wekyb3d8bbwe" -ErrorAction Stop
        Update-SessionPath
    } catch {
        Write-Host "App Installer registration did not complete: $($_.Exception.Message)"
    }

    if (Get-Command winget -ErrorAction SilentlyContinue) {
        return
    }

    Write-Host "Repairing Windows Package Manager with Microsoft.WinGet.Client."
    try {
        Install-PackageProvider -Name NuGet -Force | Out-Null
        Install-Module -Name Microsoft.WinGet.Client -Force -AllowClobber -AcceptLicense -Repository PSGallery -Scope CurrentUser -Confirm:$false | Out-Null
        Import-Module Microsoft.WinGet.Client -Force

        if (Test-IsAdministrator) {
            Repair-WinGetPackageManager -AllUsers
        } else {
            Repair-WinGetPackageManager
        }

        Update-SessionPath
    } catch {
        Write-Host "Windows Package Manager repair did not complete: $($_.Exception.Message)"
    }

    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw "winget is still unavailable. Install Microsoft App Installer / Windows Package Manager, then rerun this skill."
    }
}

function Get-PythonUserScriptsPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PythonCommand
    )

    $userScriptsPath = & $PythonCommand -c "import os, site; print(os.path.join(site.USER_BASE, 'Scripts'))"
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($userScriptsPath)) {
        throw "Could not discover the Python user Scripts directory."
    }

    return $userScriptsPath.Trim()
}

function Install-PythonIfMissing {
    $python = Get-Command python -ErrorAction SilentlyContinue
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($python -or $py) {
        return
    }

    Write-Host "Python was not found. Installing Python 3.13 with winget."
    Invoke-CommandChecked -FilePath "winget" -Arguments @(
        "install",
        "--id", "Python.Python.3.13",
        "--exact",
        "--accept-source-agreements",
        "--accept-package-agreements",
        "--disable-interactivity"
    )
    Update-SessionPath
}

function Install-WingetTool {
    param(
        [Parameter(Mandatory = $true)]
        [hashtable]$Tool
    )

    if (Test-CommandHealth -Command $Tool.Command -VersionArgs $Tool.VersionArgs) {
        Write-Host "winget tool already healthy: $($Tool.Command)"
        return
    }

    Write-Host "Installing or repairing winget package: $($Tool.Id)"
    Invoke-CommandChecked -FilePath "winget" -Arguments @(
        "install",
        "--id", $Tool.Id,
        "--exact",
        "--accept-source-agreements",
        "--accept-package-agreements",
        "--disable-interactivity"
    )
}

function Install-PipxTool {
    param(
        [Parameter(Mandatory = $true)]
        [hashtable]$Tool
    )

    $command = Get-Command $Tool.Command -ErrorAction SilentlyContinue
    if ($command) {
        Write-Host "pipx tool already available, upgrading: $($Tool.Package)"
        & pipx upgrade $Tool.Package
        if ($LASTEXITCODE -ne 0) {
            Write-Host "pipx upgrade did not complete for $($Tool.Package); leaving existing command in place."
        }
        return
    }

    Write-Host "Installing pipx package: $($Tool.Package)"
    Invoke-CommandChecked -FilePath "pipx" -Arguments @("install", $Tool.Package)
}

function Test-Tool {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,

        [string[]]$VersionArgs = @()
    )

    $resolved = Get-Command $Command -ErrorAction Stop
    Write-Host "Verified command: $Command -> $($resolved.Source)"

    if ($VersionArgs.Count -gt 0) {
        $commandLine = @($Command) + $VersionArgs
        $output = & cmd.exe /d /c ($commandLine -join " ") 2>&1
        $exitCode = $LASTEXITCODE
        $output | Select-Object -First 3 | ForEach-Object { Write-Host "  $_" }
        if ($exitCode -ne 0) {
            throw "Version check failed for command: $Command"
        }
    } elseif ($Command -eq "7z") {
        $output = & cmd.exe /d /c $Command 2>&1
        $exitCode = $LASTEXITCODE
        $output | Select-Object -First 2 | ForEach-Object { Write-Host "  $_" }
        if ($exitCode -ne 0) {
            throw "Version check failed for command: $Command"
        }
    }
}

if (-not [System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([System.Runtime.InteropServices.OSPlatform]::Windows)) {
    Write-Error "setup-codex-prerequisites is Windows-only. This script did not install or change anything. Use or create a separate macOS/Linux bootstrap for this machine."
    exit 2
}

Install-WingetIfMissing
Install-PythonIfMissing
$pythonCommand = Get-PythonCommand
$pythonUserScriptsPath = Get-PythonUserScriptsPath -PythonCommand $pythonCommand

Add-UserPathEntry -Path $pythonUserScriptsPath -Create
Add-UserPathEntry -Path (Join-Path $env:USERPROFILE ".local\bin") -Create
Add-UserPathEntry -Path "C:\Program Files\Git\cmd"
Add-UserPathEntry -Path "C:\Program Files\nodejs"
Add-UserPathEntry -Path (Join-Path $env:APPDATA "npm") -Create
Add-UserPathEntry -Path "C:\Program Files\PowerShell\7"
Update-SessionPath

Write-Host "Installing Python environment tools: uv and pipx"
Invoke-CommandChecked -FilePath $pythonCommand -Arguments @("-m", "pip", "install", "--user", "--upgrade", "uv", "pipx")
Update-SessionPath

Write-Host "Installing Python support libraries: PyYAML"
Invoke-CommandChecked -FilePath $pythonCommand -Arguments @("-m", "pip", "install", "--user", "--upgrade", "PyYAML")
Update-SessionPath

Invoke-CommandChecked -FilePath "pipx" -Arguments @("ensurepath")
Update-SessionPath

foreach ($tool in $PipxTools) {
    Install-PipxTool -Tool $tool
    Update-SessionPath
}

foreach ($tool in $WingetTools) {
    Install-WingetTool -Tool $tool
    Update-SessionPath
}

Add-UserPathEntry -Path "C:\Program Files\7-Zip"
Add-UserPathEntry -Path "C:\Program Files\Git\cmd"
Add-UserPathEntry -Path "C:\Program Files\nodejs"
Add-UserPathEntry -Path (Join-Path $env:APPDATA "npm") -Create
Add-UserPathEntry -Path "C:\Program Files\PowerShell\7"
Add-UserPathEntry -Path $pythonUserScriptsPath -Create
Update-SessionPath

Write-Host "Verifying installed commands"
Test-Tool -Command "uv" -VersionArgs @("--version")
Test-Tool -Command "uvx" -VersionArgs @("--version")
Test-Tool -Command "pipx" -VersionArgs @("--version")
Test-Tool -Command "npm" -VersionArgs @("--version")
Invoke-CommandChecked -FilePath $pythonCommand -Arguments @("-c", "import yaml; print('PyYAML ' + yaml.__version__)")

foreach ($tool in $PipxTools) {
    Test-Tool -Command $tool.Command -VersionArgs $tool.VersionArgs
}

foreach ($tool in $WingetTools) {
    Test-Tool -Command $tool.Command -VersionArgs $tool.VersionArgs
}

Write-Host "Codex prerequisite setup completed. Restart already-open terminals to inherit PATH changes."
