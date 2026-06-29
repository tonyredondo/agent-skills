#!/usr/bin/env sh
set -eu

skip_latest_check=0
for arg in "$@"; do
    case "$arg" in
        --skip-latest-check)
            skip_latest_check=1
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            echo "Usage: sh check_ytdlp.sh [--skip-latest-check]" >&2
            exit 2
            ;;
    esac
done

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

get_latest_ytdlp_release() {
    if command -v curl >/dev/null 2>&1; then
        curl --max-time 15 -fsSL \
            -H "Accept: application/vnd.github+json" \
            -H "User-Agent: codex-yt-dlp-skill" \
            "https://api.github.com/repos/yt-dlp/yt-dlp/releases/latest" |
            sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' |
            sed 's/^v//' |
            sed -n '1p'
        return
    fi

    if command -v wget >/dev/null 2>&1; then
        wget -T 15 -qO- \
            --header="Accept: application/vnd.github+json" \
            --header="User-Agent: codex-yt-dlp-skill" \
            "https://api.github.com/repos/yt-dlp/yt-dlp/releases/latest" |
            sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' |
            sed 's/^v//' |
            sed -n '1p'
        return
    fi

    return 1
}

compare_dotted_versions() {
    awk -v installed="$1" -v latest="$2" '
        BEGIN {
            installed_count = split(installed, installed_parts, ".")
            latest_count = split(latest, latest_parts, ".")
            max_count = installed_count > latest_count ? installed_count : latest_count
            for (i = 1; i <= max_count; i++) {
                installed_part = installed_parts[i] + 0
                latest_part = latest_parts[i] + 0
                if (installed_part < latest_part) {
                    print -1
                    exit
                }
                if (installed_part > latest_part) {
                    print 1
                    exit
                }
            }
            print 0
        }
    '
}

compare_ytdlp_versions() {
    installed="$1"
    latest="$2"

    if [ -z "$installed" ]; then
        echo "Version status: install yt-dlp before running download actions."
        return
    fi

    if [ "$installed" = "$latest" ]; then
        echo "Version status: installed version matches the latest GitHub release."
        return
    fi

    comparison="$(compare_dotted_versions "$installed" "$latest")"
    if [ "$comparison" -lt 0 ]; then
        echo "Version status: update recommended before using yt-dlp."
        echo "Update through the same route used for installation."
    elif [ "$comparison" -gt 0 ]; then
        echo "Version status: installed version is newer than the latest stable release."
    else
        echo "Version status: installed version differs from the latest GitHub release."
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
show_command curl
show_command wget

echo
echo "Runtime versions"
show_version_line yt-dlp yt-dlp --version
show_version_line ffmpeg ffmpeg -version
show_version_line ffprobe ffprobe -version
show_version_line python3 python3 --version
show_version_line deno deno --version
show_version_line node node --version
show_version_line qjs qjs -v

if [ "${YT_DLP_SKIP_LATEST_CHECK:-}" = "1" ] || [ "$skip_latest_check" -eq 1 ]; then
    echo
    echo "yt-dlp release check"
    echo "Skipped by --skip-latest-check or YT_DLP_SKIP_LATEST_CHECK=1."
else
    echo
    echo "yt-dlp release check"
    installed_ytdlp_version="$(yt-dlp --version 2>/dev/null || true)"
    if [ -n "$installed_ytdlp_version" ]; then
        echo "Installed version: $installed_ytdlp_version"
    else
        echo "Installed version: not found"
    fi

    latest_ytdlp_version="$(get_latest_ytdlp_release 2>/dev/null || true)"
    if [ -n "$latest_ytdlp_version" ]; then
        echo "Latest GitHub release: $latest_ytdlp_version"
        compare_ytdlp_versions "$installed_ytdlp_version" "$latest_ytdlp_version"
    else
        echo "Latest GitHub release: unavailable; network, GitHub API access, curl, or wget may be missing."
    fi
fi

echo
echo "PATH"
printf '%s\n' "$PATH" | tr ':' '\n'
