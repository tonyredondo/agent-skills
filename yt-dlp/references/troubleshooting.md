# yt-dlp Troubleshooting

## First Diagnostic Pass

```powershell
yt-dlp --version
yt-dlp -Uv "<URL>"
yt-dlp -F "<URL>"
Get-Command yt-dlp -All
Get-Command ffmpeg -All
ffmpeg -version
ffprobe -version
```

Use `-Uv` for issue diagnostics because it combines update information and verbose output. Redact or summarize logs before sharing them; URLs, headers, cookies, paths, and account details may appear.

## Update Problems

Update through the same route used for installation. For complete platform-specific commands, read `installation.md`.

```powershell
winget upgrade -e --id yt-dlp.yt-dlp
python -m pip install -U "yt-dlp[default]"
yt-dlp -U
```

Use `yt-dlp -U` for standalone release binaries. If stable is broken because a site changed, the official README recommends trying nightly before filing a bug:

```powershell
yt-dlp --update-to nightly
```

Be careful with updates from arbitrary repositories; official documentation notes there is no binary verification for `--update-to owner/repo` targets.

## Common Failures

- `yt-dlp` not recognized: restart the shell, inspect PATH, and check the package manager install.
- Format unavailable: run `yt-dlp -F "<URL>"`; format IDs are extractor-specific and may change.
- Merge or conversion failure: update ffmpeg, check `ffprobe`, and pass `--ffmpeg-location` if PATH resolves the wrong binary.
- Auth or age-gated content failure: use only the user's own cookies with `--cookies-from-browser <browser>` or `--cookies cookies.txt`.
- Cookie failure: close the browser if the profile is locked, confirm the right browser/profile, and treat cookie files as sensitive.
- Playlist unexpectedly downloads many items: add `--no-playlist`, `--playlist-items`, `--max-downloads`, or `--download-archive`.
- Repeated batch downloads: add `--download-archive archive.txt`.
- Bad filenames on Windows: use `--windows-filenames`, `--trim-filenames`, a shorter `-o`, and a simple `-P`.
- TLS errors: avoid `--no-check-certificates` unless the user understands the risk and there is a specific certificate problem.
- Partial site breakage: update yt-dlp first, then retry with `-Uv`.
