#!/usr/bin/env sh
set -eu

show_command() {
    name="$1"
    path="$(command -v "$name" 2>/dev/null || true)"
    if [ -n "$path" ]; then
        printf '%-8s found: %s\n' "$name" "$path"
    else
        printf '%-8s not found\n' "$name"
    fi
}

show_version_line() {
    name="$1"
    shift
    if command -v "$name" >/dev/null 2>&1; then
        printf '%s: ' "$name"
        "$@" 2>&1 | sed -n '1p'
    fi
}

echo "Command resolution"
show_command yt-dlp
show_command ffmpeg
show_command ffprobe
show_command python3
show_command deno
show_command node
show_command qjs

echo
echo "Runtime versions"
show_version_line yt-dlp yt-dlp --version
show_version_line ffmpeg ffmpeg -version
show_version_line ffprobe ffprobe -version
show_version_line python3 python3 --version
show_version_line deno deno --version
show_version_line node node --version
show_version_line qjs qjs -v

echo
echo "PATH"
printf '%s\n' "$PATH" | tr ':' '\n'
