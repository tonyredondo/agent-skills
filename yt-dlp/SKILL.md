---
name: yt-dlp
description: Use yt-dlp from Codex for cross-platform installation/setup, command-line audio/video downloads, media metadata inspection, format selection, playlists, audio extraction, subtitles, cookies/browser authentication, output templates, configuration files, ffmpeg post-processing, update checks, and troubleshooting on Windows, macOS, Linux, Android/Termux, PowerShell, Bash, or Zsh. Trigger when the user asks to install, update, configure, download or inspect media with yt-dlp, write yt-dlp commands, diagnose yt-dlp/ffmpeg/path/runtime errors, or convert a download request into safe CLI steps.
---

# yt-dlp CLI

## Core Workflow

Start with inspection unless the user explicitly gives a complete command and asks to run it:

1. Read `references/installation.md` when yt-dlp is missing, stale, or needs installation on Windows, macOS, Linux, Android/Termux, or direct binary/pip routes.
2. Verify the environment with `scripts/check_ytdlp.ps1` on Windows or `scripts/check_ytdlp.sh` on macOS/Linux when setup, PATH, ffmpeg, ffprobe, Python, or JavaScript runtime state matters.
3. Use `yt-dlp --version` and `yt-dlp --help` for exact local behavior when options may have changed.
4. Use `yt-dlp --simulate`, `--dump-json`, `--print`, or `--list-formats` before downloading when choosing formats, naming files, or diagnosing extractor behavior.
5. Build the smallest command that satisfies the request. Quote URLs and output templates.
6. Prefer `--download-archive` for channels, playlists, recurring jobs, or batch work to avoid repeated downloads.
7. Verify output paths and generated filenames after any real download.

Do not bypass site access controls, DRM, paywalls, private content restrictions, or user account boundaries. If cookies are needed, use the user's own browser profile via `--cookies-from-browser` or a user-provided cookies file; never ask for or expose credentials.

## Official Sources

Use the current official documentation for details before advanced work:

- Repository and README: https://github.com/yt-dlp/yt-dlp
- Installation wiki: https://github.com/yt-dlp/yt-dlp/wiki/Installation
- FAQ wiki: https://github.com/yt-dlp/yt-dlp/wiki/FAQ
- Supported sites list: https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md

The README contains the main option reference, including usage/options, configuration, output templates, format selection, metadata, extractor arguments, plugins, embedding, and differences from youtube-dl.

## Common Tasks

Read `references/command-patterns.md` when the user needs concrete commands for:

- Listing formats and selecting video/audio quality.
- Downloading single videos, playlists, channels, or batches.
- Extracting audio, remuxing, recoding, embedding thumbnails, and using ffmpeg.
- Writing output templates and safe Windows paths.
- Downloading subtitles or metadata.
- Using cookies from a browser profile.
- Creating or editing a yt-dlp configuration file.
- Troubleshooting extractor, ffmpeg, PATH, stale version, or filename errors.

Read `references/installation.md` when the user asks to install, update, repair, or choose an install method for any platform.

Read `references/windows-powershell.md` for Windows quoting, PATH, ffmpeg resolution, and filename guidance.

Read `references/troubleshooting.md` when errors involve stale extractors, cookies, ffmpeg, format selection, playlists, TLS, or verbose diagnostic logs.

## Install Notes

Choose the install route that matches the user's OS and package manager. Prefer the official release binary or the platform package manager the user already uses. Package-manager builds may be maintained by third parties and can lag behind the official release; when an extractor breaks, verify the local version and update through the same route used for installation.

For full install and update commands, read `references/installation.md`.

## Windows Notes

When several `ffmpeg.exe` files exist, inspect `Get-Command ffmpeg -All`. A non-yt-dlp ffmpeg earlier in PATH may be used by default. If the intended ffmpeg is not first, either pass `--ffmpeg-location <bin-dir>` in the yt-dlp command or fix PATH ordering outside the download command.

## Running Commands

Before running a real download, confirm the target URL and destination are appropriate. For large playlists or uncertain formats, start with one of:

```powershell
yt-dlp --simulate --print "%(title)s [%(id)s].%(ext)s" "<URL>"
yt-dlp --list-formats "<URL>"
yt-dlp --dump-json "<URL>"
yt-dlp -Uv "<URL>"
```

Use `--verbose` only for diagnostics. Avoid pasting verbose logs into final answers when they may contain URLs, cookies, paths, or account-identifying details; summarize relevant lines instead.

## Updating

Update through the same route used for installation. For standalone release binaries, `yt-dlp -U` is supported. The official README notes stable, nightly, and master release channels; use nightly only when a current stable release appears broken by site changes or the user asks for it.
