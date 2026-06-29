# yt-dlp Command Patterns

Use these as starting points. Check `yt-dlp --help` or the official README when an option's behavior matters.

## Inspect Before Downloading

```powershell
yt-dlp --version
yt-dlp --list-formats "<URL>"
yt-dlp --simulate --print "%(title)s [%(id)s].%(ext)s" "<URL>"
yt-dlp --dump-json "<URL>"
yt-dlp -Uv "<URL>"
```

Use `--dump-json` for machine-readable metadata and `--print` for lightweight fields. Useful print fields include `title`, `id`, `ext`, `duration`, `upload_date`, `uploader`, `webpage_url`, `format_id`, `resolution`, `vcodec`, and `acodec`.

## Basic Downloads

```powershell
yt-dlp "<URL>"
yt-dlp -P "$env:USERPROFILE\Downloads\yt-dlp" "<URL>"
yt-dlp -o "%(uploader)s\%(title).200B [%(id)s].%(ext)s" "<URL>"
yt-dlp --windows-filenames --trim-filenames 180 "<URL>"
```

Use `%(title).200B` or another length limit for long titles. Include `%(id)s` when duplicate titles are possible. Quote templates because `%`, spaces, and brackets are easy to mishandle. Do not hard-code an extension like `.mp4` in `-o`; use `%(ext)s`.

## Format Selection

```powershell
yt-dlp -F "<URL>"
yt-dlp -f "bv*+ba/b" "<URL>"
yt-dlp -S "res:1080,codec:h264:m4a" "<URL>"
yt-dlp -S "res,ext:mp4:m4a" "<URL>"
yt-dlp -f "bv*[height<=720]+ba/b[height<=720]" "<URL>"
yt-dlp -f "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b" "<URL>"
yt-dlp --merge-output-format mp4 "<URL>"
```

Prefer `-S` for quality sorting and constraints when possible. Use `-f` when selecting exact format combinations or filters. Separate video/audio formats usually require ffmpeg for merging.

## Audio Extraction

```powershell
yt-dlp -x --audio-format mp3 --audio-quality 0 "<URL>"
yt-dlp -x --audio-format m4a "<URL>"
yt-dlp -x --embed-thumbnail --add-metadata "<URL>"
```

Audio extraction and embedding typically require ffmpeg/ffprobe. If ffmpeg is not found, run the diagnostic script or pass `--ffmpeg-location`.

## Playlists, Channels, and Batches

```powershell
yt-dlp --yes-playlist "<PLAYLIST_URL>"
yt-dlp --playlist-items 1:10 "<PLAYLIST_URL>"
yt-dlp --download-archive ".\archive.txt" "<CHANNEL_OR_PLAYLIST_URL>"
yt-dlp -a ".\urls.txt" --download-archive ".\archive.txt"
```

Use `--download-archive` for any repeated run. Use `--no-playlist` when the user provided a video URL that also belongs to a playlist but wants only one item.

## Subtitles and Metadata

```powershell
yt-dlp --write-subs --sub-langs "en.*,es.*" --convert-subs srt "<URL>"
yt-dlp --list-subs "<URL>"
yt-dlp --write-auto-subs --sub-langs "en" --skip-download "<URL>"
yt-dlp --write-info-json --write-thumbnail "<URL>"
yt-dlp --embed-subs --embed-thumbnail --add-metadata "<URL>"
```

Auto-generated subtitles are different from creator-provided subtitles. Use `--sub-langs "all,-live_chat"` when live chat should be excluded. Embedding subtitles or thumbnails may require compatible containers and ffmpeg.

## Cookies and Authenticated Sites

```powershell
yt-dlp --cookies-from-browser firefox "<URL>"
yt-dlp --cookies-from-browser chrome "<URL>"
yt-dlp --cookies ".\cookies.txt" "<URL>"
yt-dlp --cookies-from-browser chrome --cookies ".\cookies.txt"
```

Use only the user's own authenticated browser profile or user-provided cookies. Do not request passwords, session tokens, or cookie contents in chat. Some browsers/profiles need the browser closed before cookies can be read.

## Post-processing and ffmpeg

```powershell
yt-dlp --remux-video mp4 "<URL>"
yt-dlp --recode-video mp4 "<URL>"
yt-dlp --embed-metadata --embed-thumbnail --embed-subs "<URL>"
yt-dlp --split-chapters "<URL>"
yt-dlp --ffmpeg-location "C:\Path\To\ffmpeg\bin" "<URL>"
```

Prefer remuxing over recoding when only the container needs to change; recoding is slower and can reduce quality. Use `--merge-output-format` for merged output containers. Cutting is not always frame-exact unless re-encoding is used.

## Configuration Files

A user config can hold defaults such as output path, archive file, and format preferences. Keep one option per line and comments with `#`.

Example Windows-oriented config:

```text
-P C:\Users\<User>\Downloads\yt-dlp
-o %(uploader)s\%(title).200B [%(id)s].%(ext)s
--download-archive C:\Users\<User>\Downloads\yt-dlp\archive.txt
-S res:1080,codec:h264:m4a
--embed-thumbnail
--add-metadata
```

Do not put secrets, passwords, or raw cookie values in a config file unless the user explicitly manages that file and understands the risk.

## Troubleshooting Checklist

- `yt-dlp` not recognized: restart the shell, check `Get-Command yt-dlp`, and verify the install method.
- Stale or broken extractor: run `yt-dlp --version`, update via the install method, then retry with `--verbose`.
- Format unavailable: run `yt-dlp -F "<URL>"` and choose from actual listed formats.
- Merge/conversion failure: check `ffmpeg -version`, `ffprobe -version`, and `Get-Command ffmpeg -All`.
- Filename/path failure on Windows: shorten templates, avoid reserved characters, and set a simple `-P` destination.
- Auth failure: use `--cookies-from-browser <browser>` from the user's profile, or retry after logging in through the browser.
- Playlist too large: use `--playlist-items`, `--max-downloads`, or `--download-archive`.
