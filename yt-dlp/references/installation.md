# yt-dlp Installation and Updates

Use this reference when yt-dlp needs to be installed, updated, repaired, or selected for a platform. Commands are based on the official yt-dlp installation wiki and README.

## General Rules

- Prefer the user's existing package manager when it is current enough.
- Prefer official release binaries when package-manager builds lag behind site changes.
- Update through the same route used for installation.
- Verify with `yt-dlp --version`.
- Supported Python is CPython 3.10+ or PyPy 3.11+.
- For post-processing, merging separate audio/video, audio extraction, thumbnails, or subtitles, install the `ffmpeg` and `ffprobe` binaries and verify with `ffmpeg -version` and `ffprobe -version`.
- For full YouTube support, ensure `yt-dlp-ejs` and a supported JavaScript runtime are available. Deno is recommended and enabled by default.
- Restart the shell after installing or changing PATH.

## Official Release Binaries

Windows x64 release binary:

```powershell
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\bin" | Out-Null
Invoke-WebRequest -Uri "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp.exe" -OutFile "$env:USERPROFILE\bin\yt-dlp.exe"
```

Other official release variants exist for Windows ARM64/x86 and Linux standalone builds. Use the current README release files table when CPU architecture or Python availability is not standard.

Linux/BSD:

```bash
mkdir -p ~/.local/bin
curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp -o ~/.local/bin/yt-dlp
chmod a+rx ~/.local/bin/yt-dlp
```

macOS:

```bash
mkdir -p ~/.local/bin
curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_macos -o ~/.local/bin/yt-dlp
chmod a+rx ~/.local/bin/yt-dlp
```

Update release binaries:

```bash
yt-dlp -U
```

On Windows, ensure the target directory, such as `%USERPROFILE%\bin`, is in PATH before relying on `yt-dlp` from a new shell. Unpackaged `.zip` variants are not the normal auto-updating release binary route.

## Python / pip

Use the default extras unless there is a reason to avoid optional dependencies. The default dependency group includes `yt-dlp-ejs`, which is needed for full YouTube support:

```bash
python3 -m pip install -U "yt-dlp[default]"
```

On Windows, the Python launcher may be required:

```powershell
py -m pip install -U "yt-dlp[default]"
```

Install without optional dependencies:

```bash
python3 -m pip install --no-deps -U yt-dlp
```

Nightly via pip:

```bash
python3 -m pip install -U --pre "yt-dlp[default]"
```

Update pip installs by rerunning the install command.

The official installation wiki does not provide a canonical pipx command. If using pipx, preserve the same principle as pip: install with the `default` extra so `yt-dlp-ejs` is installed and upgraded with yt-dlp.

## Windows Package Managers

winget:

```powershell
winget install -e --id yt-dlp.yt-dlp
winget upgrade -e --id yt-dlp.yt-dlp
```

The official wiki also lists the short form:

```powershell
winget install yt-dlp
winget upgrade yt-dlp
```

Scoop:

```powershell
scoop install yt-dlp
scoop update yt-dlp
```

Chocolatey:

```powershell
choco install yt-dlp
choco upgrade yt-dlp
```

After installing on Windows, inspect command resolution:

```powershell
Get-Command yt-dlp -All
Get-Command ffmpeg -All
Get-Command ffprobe -All
```

## macOS

Homebrew:

```bash
brew install yt-dlp
brew upgrade yt-dlp
```

MacPorts:

```bash
sudo port install yt-dlp
sudo port selfupdate
sudo port upgrade yt-dlp
```

The official macOS release binary is also available as `yt-dlp_macos`; see "Official Release Binaries" above.

## Linux

Homebrew on Linux:

```bash
brew install yt-dlp
brew upgrade yt-dlp
```

Arch Linux / pacman:

```bash
sudo pacman -Syu yt-dlp
```

Ubuntu/Debian-family PPA route listed by the official wiki:

```bash
sudo add-apt-repository ppa:tomtomtom/yt-dlp
sudo apt update
sudo apt install yt-dlp
```

Update the PPA install:

```bash
sudo apt update
sudo apt install yt-dlp
```

Snap:

```bash
sudo snap install --edge yt-dlp
sudo snap refresh --edge yt-dlp
```

Alpine Linux:

```sh
doas apk -U add yt-dlp
doas apk -U upgrade yt-dlp
```

Alpine core-only package:

```sh
doas apk -U add yt-dlp-core
```

On postmarketOS, `sudo` may be needed instead of `doas`.

## Android / Termux

```bash
termux-setup-storage
pkg update && pkg upgrade
pkg install python python-pip
pip install -U "yt-dlp[default]"
pkg install ffmpeg
```

Update:

```bash
pip install -U "yt-dlp[default]"
```

## ffmpeg Notes

yt-dlp can download many formats without ffmpeg, but ffmpeg/ffprobe are strongly recommended for merging, remuxing, recoding, audio extraction, embedding thumbnails/subtitles, and metadata workflows.

Install the ffmpeg binary, not the Python package named `ffmpeg`.

Use the platform package manager when practical:

```bash
brew install ffmpeg
sudo pacman -Syu ffmpeg
sudo apt install ffmpeg
pkg install ffmpeg
```

On Windows, `winget install -e --id yt-dlp.yt-dlp` may install a yt-dlp-specific FFmpeg dependency. If a different `ffmpeg.exe` appears first in PATH, pass `--ffmpeg-location <bin-dir>` or adjust PATH ordering.

## JavaScript Runtime and EJS

YouTube downloads may require external JavaScript challenge solving through `yt-dlp-ejs` plus a supported JavaScript runtime.

- Deno is recommended and enabled by default.
- Node can be used with `--js-runtimes node`.
- QuickJS can be used with `--js-runtimes quickjs`.
- Bun is deprecated in the official EJS wiki; avoid choosing it for new setup unless the user specifically asks.

Official bundled executables and the Unix zipimport binary include EJS scripts. PyPI installs should use `yt-dlp[default]`. Third-party package managers may or may not bundle current EJS scripts; if YouTube fails after yt-dlp is current, check the official EJS wiki before adding `--remote-components`.

## Staleness and Nightly

Package-manager builds may lag behind the official release. If a site extractor breaks:

1. Run `yt-dlp --version`.
2. Update through the same install route.
3. Retry with `yt-dlp -Uv "<URL>"`.
4. Try nightly only when stable appears broken or the user asks for it:

```bash
yt-dlp --update-to nightly
```

Be careful with arbitrary `--update-to owner/repo` targets because official documentation notes binary verification is not provided for binaries from other repositories.
