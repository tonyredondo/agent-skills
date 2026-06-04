# Windows and PowerShell Notes

## Command Resolution

```powershell
Get-Command yt-dlp -All
Get-Command ffmpeg -All
Get-Command ffprobe -All
where.exe yt-dlp
where.exe ffmpeg
```

Restart PowerShell after installing with `winget`, Scoop, Chocolatey, pipx, or after editing PATH. If several `ffmpeg.exe` files exist, the first one in PATH wins. Use `--ffmpeg-location "C:\path\to\ffmpeg\bin"` when yt-dlp should use a specific ffmpeg.

## Quoting

Always quote URLs and paths:

```powershell
yt-dlp "https://example.test/watch?v=id&list=listid"
yt-dlp -P "D:\Media Downloads" "<URL>"
```

If an ID begins with `-`, pass `--` before the ID or use the full URL:

```powershell
yt-dlp -- -wNyEUrxzFU
```

## Output Templates

Use `%(ext)s` in output templates instead of forcing a suffix:

```powershell
yt-dlp -o "%(uploader)s\%(title).200B [%(id)s].%(ext)s" "<URL>"
```

For Windows-safe names, add:

```powershell
yt-dlp --windows-filenames --trim-filenames 180 "<URL>"
```

In `.bat` or `.cmd` files, double percent signs in templates, for example `%%(title)s.%%(ext)s`. In PowerShell commands, keep single percent signs.

## Windows Installation

For Windows install and update commands, read `installation.md`. Keep this file focused on PowerShell quoting, PATH resolution, ffmpeg selection, and Windows filename behavior.
